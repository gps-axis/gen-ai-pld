#!/usr/bin/env python3
"""Step 4+5 - grade every candidate against the CLEANED source and pick.

    python tools/grade_flats.py --run runs/<stamp>

Answers "which generated flat should we ship", scoring shape and wrinkles, and
checking the generation did not quietly redraw the garment's construction.
Replaces measure.py and the by-eye contact-sheet review that used to follow it.

Why it is built this way
  Instruct-edit models re-synthesize the subject rather than editing pixels, so
  every candidate is a fresh chance to invent a seam, shift a neckline or
  repaint a colourway. Invented detail is usually plausible, so it survives
  casual review. The grade therefore measures FIDELITY to the cleaned source -
  is this still the same garment - and not presentation polish.

  That is a correction, and an expensive one. The grade used to be 45% shape +
  45% wrinkles + 10% background, with wrinkles ranked WITHIN the batch. On
  runs/20260819_205617 that put the plastic-smooth redraws at the top (79.1 and
  71.4) and the faithful re-lays at the bottom (16.6 and 18.8) - exactly upside
  down against both the contact sheet and stage 3. Two faults did it: a
  batch-relative smoothness term rewards whichever candidate ironed the fabric
  hardest, and nothing in the formula compared the candidate to the source at
  all.

Stages
  1. Deterministic measurement against the cleaned source, no model:
       silhouette IoU   the candidate's outline against the source's, both
                        normalised to their own bounding boxes so a legitimate
                        rescale or re-centre costs nothing. 35% of the grade.
       colour drift     dE76 between the two garment colours. 20%.
       wrinkle delta    distance from the SOURCE's own texture energy, not from
                        the batch's smoothest. 20%.
       symmetry         the silhouette against its own mirror. 15%.
       background       backdrop lightness, measured. 10%.
     Free, repeatable, cannot hallucinate.
  2. Vision grading, --votes independent calls per candidate. ADVISORY ONLY -
     see --judge. Both model modes measured badly on this project (one
     saturated at 100/100, the other picked by slot), so they no longer feed
     the grade; they print beside it.
  3. Construction check on native-resolution crops against the reference. Any
     MISMATCH marks the candidate REJECT *and* costs it a fixed penalty per
     altered region, so the number and the label finally say the same thing -
     a run once had to arbitrate between a 79.1 grade and three MISMATCH flags
     on the same image, and spent thirty turns doing it. Which regions get
     cropped follows the garment: the profile is read from the garment_type in
     <run>/reference_selection.json, so grading a bra with the leggings bands is
     not something a forgotten flag can cause.

Three notes carried over, because they were paid for here and the code does not
encode them:

  * The reference MUST be <run>/archive/offset_upload.jpg, the cleaned image the
    generator actually received - never inputs/off_set_image.jpg. The raw input
    still carries the hang tag, and a construction check run against it reports
    the correctly-removed tag as "a label removed" on every candidate. That is
    why --reference defaults off the run folder and not off inputs/.

  * Anything the CLEAN step legitimately removed - pins, clips, a tag - is a
    difference stage 3 will find and call a defect unless it is told. Pass
    --expected-changes "pearl-headed pins removed" and it is declared as
    correct. On the run this was written for, three pearl-headed pins survived
    into the source, every candidate correctly dropped them, and all ten came
    back MISMATCH on every region for it.

  * Every measurement here is taken inside common.py's garment mask, which
    finds a pale garment by CHROMA rather than by brightness. The luminance
    rule it replaced collapsed to a 3% strip of frame on three candidates of
    that same run, and the metrics taken off it were confident nonsense.

ImageMagick + sips + the project venv. Vision helpers come from vision.py.

Usage
  python tools/grade_flats.py --run runs/<stamp>
  python tools/grade_flats.py --run runs/<stamp> --expected-changes "pins removed"
  python tools/grade_flats.py --run runs/<stamp> --no-construction
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import common as C
from vision import (
    CACHE, Client, ensure_small, image_part, parse_json_blob, settled,
    text_part, transient, DEFAULT_BASE_URL, DEFAULT_MODEL,
)

ROOT = Path(__file__).resolve().parent.parent
WORK = CACHE / "grade"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
# Every image is rescaled so the garment bbox is this tall before texture
# is measured, so wrinkle energy compares fabric, not source resolution.
NORM_BBOX_H = 900

RES_ORDER = {"1K": 1, "2K": 2, "4K": 3}

# What the grade is made of. Fidelity to the source carries 75% of it between
# silhouette, colour and texture, because those are the three ways a
# re-synthesised garment stops being the product while still photographing
# well. Symmetry and background are presentation and are what is left.
WEIGHTS = {"silhouette": 0.35, "colour": 0.20, "wrinkle": 0.20,
           "symmetry": 0.15, "background": 0.10}

# What stage 3 asks. Part of every stored verdict's fingerprint, so adding a
# question re-judges rather than reusing answers to the old one.
#   1  construction per region
#   2  + orientation, on the profile's face region
STAGE3_VERSION = 2


def auto_select(arch: Path):
    """Every generated image at the highest resolution present.

    Carried over from measure.py. A probe generated at the same resolution as
    the final batch is a candidate - same model, same prompt, same pixels.
    Excluding it by filename discarded the three best images of a real run.
    """
    sf = arch / "seeds.json"
    try:
        man = json.loads(sf.read_text()) if sf.exists() else {}
    except json.JSONDecodeError:
        man = {}
    man = {k: v for k, v in man.items() if isinstance(v, dict)}
    if not man:
        return sorted(arch.glob("cand_*.png")), "cand_*.png"
    best = max(RES_ORDER.get(v.get("resolution"), 0) for v in man.values())
    keep = [k for k, v in man.items() if RES_ORDER.get(v.get("resolution"), 0) == best]
    res = next(v["resolution"] for v in man.values()
               if RES_ORDER.get(v.get("resolution"), 0) == best)
    paths = sorted(p for p in (arch / f"{k}.png" for k in keep) if p.exists())
    dropped = len(man) - len(keep)
    print(f"grading {len(paths)} image(s) at {res}"
          + (f"; {dropped} lower-resolution probe(s) held back" if dropped else ""))
    return paths, f"images at {res}"


# --------------------------------------------------------------------------
# Stage 1 - deterministic metrics
# --------------------------------------------------------------------------

def _fx(args: list[str]) -> float:
    out = subprocess.run(["magick", *args], check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def backdrop(src: Path, width: int = 800) -> dict:
    """Lightness and flatness of the plate, from a corner of the frame.

    Deliberately mask-free: it is the one measurement that must keep working
    when the garment cannot be found at all. Measured rather than judged - the
    vision model rated a visibly grey backdrop 100/100.
    """
    crop = [str(src), "-resize", f"{width}x", "-gravity", "northwest",
            "-crop", "60x60+0+0", "+repage", "-colorspace", "Gray"]
    return {"bg_sd": round(_fx(crop + ["-format", "%[fx:standard_deviation]", "info:"]), 4),
            "bg_lum": round(_fx(crop + ["-format", "%[fx:mean]", "info:"]), 4)}


def measure(src: Path, ref: dict | None = None,
            ref_mask=None) -> tuple[dict, "object"]:
    """Everything stage 1 knows about one image, and the mask it was taken over.

    `ref` is the reference's own measurement dict and `ref_mask` its mask; pass
    neither for the reference itself. With them, the three fidelity numbers
    appear - silhouette IoU, colour drift and the wrinkle delta - because all
    three are differences and none of them means anything about a single image
    on its own.

    Every number here is taken inside common.py's garment mask, so the `cue`
    field is part of the record: a measurement is only as good as the pixels it
    was taken over, and this batch is why that sentence is in the file.
    """
    e = C.garment_evidence(src)
    m = e["mask"]
    out = {
        "cue": e["cue"],
        "coverage": round(e["area"], 4),
        "aspect": e["aspect"],
        "bbox": e["bbox"],
        "chroma_threshold": e["chroma_threshold"],
        "chroma_inside": e["chroma_inside"],
        "luma_contrast": e["luma_contrast"],
        **backdrop(src),
    }
    if not m.any():
        return {**out, "symmetry": 0.0, "asymmetry": 1.0, "wrinkle_energy": 0.0,
                "rgb": [0.0, 0.0, 0.0], "found": False}, m

    sym = C.shape_stats(m)["symmetry"]
    rgb = C.garment_rgb(src, m)
    out.update({
        "found": True,
        "symmetry": round(float(sym), 4),
        # Kept under its old name so a reader comparing two runs is comparing
        # the same quantity; it is now measured on the chroma mask.
        "asymmetry": round(1.0 - float(sym), 4),
        "wrinkle_energy": round(C.wrinkle_energy(src, m, NORM_BBOX_H), 4),
        "rgb": [round(float(v), 1) for v in rgb],
    })
    if ref and ref.get("found") and ref_mask is not None:
        drift = C.colour_drift(rgb, ref["rgb"])
        out.update({
            "silhouette_iou": round(C.silhouette_iou(m, ref_mask), 4),
            "colour_de": round(drift["de"], 3),
            "colour_drgb": round(drift["drgb"], 3),
            "wrinkle_delta": round(out["wrinkle_energy"] - ref["wrinkle_energy"], 4),
        })
    return out, m


# --------------------------------------------------------------------------
# Stage 2 - vision grading
# --------------------------------------------------------------------------

GRADE_PROMPT = """Image 1 is the REFERENCE: the real garment, photographed flat as-is.
Image 2 is a CANDIDATE: a generated ecommerce flat of that same garment.

The candidate is SUPPOSED to differ from the reference in these ways. None of these
is a defect, do not deduct for them:
  - cleaner / whiter background, softer or removed shadows
  - the garment straightened, re-centred, re-framed or rescaled
  - creases and rumples relaxed - that is the entire point of the edit
  - hangtags, pins, props or clips removed
%EXPECTED%

Grade the CANDIDATE as an ecommerce product flat, on two axes.

SHAPE - is the silhouette right for a product listing?
  100 = both sides symmetric and evenly laid, legs/arms straight and parallel,
        natural garment proportions, nothing stretched, bent, pinched or warped
   50 = noticeably lopsided, one side wider or longer, a leg bowed or twisted
    0 = distorted, melted, impossible geometry, garment unrecognisable

WRINKLES - how clean is the surface?
  100 = smooth and evenly lit, reads as pressed, only soft structural shading
   50 = several visible creases or blotchy shading across panels
    0 = heavily rumpled, harsh fold shadows everywhere

Return ONE JSON object, nothing else:
{"shape": <0-100 integer>,
 "shape_issues": "<the single worst shape problem, or 'none'>",
 "wrinkles": <0-100 integer>,
 "wrinkle_issues": "<the single worst wrinkle problem, or 'none'>",
 "background": <0-100 integer, how clean and even the backdrop is>,
 "verdict": "ship" | "borderline" | "reject"}"""

CONSTRUCTION_PROMPT = """Both crops show the SAME REGION ({region}) of the SAME garment.
Image 1 is the REFERENCE (the real product). Image 2 is a GENERATED version.

A generative model redrew image 2, so it may have invented, moved or deleted
construction detail. That is what you are looking for, and only that.

Expected differences that are NOT discrepancies: background colour, brightness,
shadow softness, scale, rotation, position in frame, overall sharpness, and the
fabric lying flatter or smoother.
%EXPECTED%
Report ONLY genuine construction differences: seams that were added or removed,
topstitching that appeared or vanished, a pocket or its opening moved or resized,
a waistband changed in height or structure, a hem or cuff changed in shape,
a logo/label added, removed or altered.
%FACE%
Return ONE JSON object, nothing else:
{{"verdict": "MATCH" | "MISMATCH",
  "detail": "<the specific construction difference, or 'none'>"%FACEKEY%}}"""

# A flip is invisible to every other test in the pipeline. The silhouette is
# near-symmetric, the colour is identical, the texture is identical, and every
# seam can be individually correct while the garment shows its reverse face. On
# runs/20260819_223347 the agent's own prompt said "the garment is shown from
# the back" and the model did as it was told; stage 3 then reported
# topstitching that had "appeared", which was the other face's real stitching,
# seen for the first time.
#
# This is asked of ONE image at a time, and the answers are compared afterwards.
# The obvious cheaper design - fold "is this the same face as image 1?" into a
# construction call that shows both crops - was built first and measured blind:
# handed a candidate MIRRORED left-for-right, it answered SAME, with a confident
# sentence about the exterior construction being visible in both. Shown the two
# images together the model reconciles them; asked to describe one image it
# observes it. The same positive control run against the question below flips
# its answer, which is the only reason this one is trusted.
FACE_PROBE = """This is a flat product photograph of one garment.

Are you looking at the OUTSIDE of the garment (the face a customer sees when it
is worn) or the INSIDE (the lining, seam allowances, interior elastic, care
labels - the face against the body)?

Then, separately: is the side facing you the FRONT of the garment or the BACK?

Return ONE JSON object, nothing else:
{"face": "OUTSIDE" | "INSIDE" | "CANNOT TELL",
 "side": "FRONT" | "BACK" | "CANNOT TELL",
 "why": "<one short sentence naming what told you>"}"""


def face_of(client: Client, small: Path) -> dict:
    """Which face and which side of the garment this image shows."""
    try:
        v = parse_json_blob(client.chat(
            [image_part(small), text_part(FACE_PROBE)],
            max_tokens=250, temperature=0.0))
    except Exception as e:  # noqa: BLE001 - a dropped probe is not a flip
        return {"face": "CANNOT TELL", "side": "CANNOT TELL",
                "why": f"probe failed: {type(e).__name__}"}
    return {"face": str(v.get("face", "CANNOT TELL")).strip().upper(),
            "side": str(v.get("side", "CANNOT TELL")).strip().upper(),
            "why": str(v.get("why", ""))[:200]}


# Asked of the vision model rather than measured, because the geometry does not
# separate. See the removed-detector note in common.py: the duplicate usually
# OVERLAPS the original, so it is one connected component, and a backdrop
# gradient makes phantom ones. A model looking at the picture answers this
# immediately.
#
# The wording works hard on the partial case. The full stacked copies are easy
# and were never the ones that shipped; the ones that got through were a collar
# and a pair of shoulders fading out at the top edge, which a model will happily
# call "one garment, slightly cropped" unless it is told that a fragment counts.
COUNT_PROBE = """This is meant to be a product photograph of ONE garment on a
plain background.

Count the garments. Include any PARTIAL or FADED one: a second collar, neckline,
shoulder, sleeve or cuff appearing anywhere in the frame counts as another
garment even if it is blurred, cut off by the edge, semi-transparent, or clearly
a duplicate of the main one. A garment overlapping or sitting behind the main
garment still counts.

Do not count the garment's own parts - its two sleeves, its neckband, its hem -
as separate garments. One sweater with two sleeves is ONE garment.

Return ONE JSON object, nothing else:
{"garments": <integer>,
 "extra": "<what the additional garment or fragment is and where, or empty>",
 "why": "<one short sentence>"}"""


def count_garments(client: Client, small: Path) -> dict:
    """How many garments are in this frame? More than one is a hard reject.

    The failure this exists for shipped in three consecutive runs - a second
    sweater stacked above the first, or the top of one fading in - because
    every geometric metric is computed on the largest connected blob and the
    ghost was discarded before anything could measure it.
    """
    try:
        v = parse_json_blob(client.chat(
            [image_part(small), text_part(COUNT_PROBE)],
            max_tokens=250, temperature=0.0))
        n = int(v.get("garments", 1))
    except Exception as e:  # noqa: BLE001 - a dropped probe must not reject
        # Fails OPEN on purpose. A probe that errors is a broken server, not a
        # broken image, and rejecting every candidate on an outage would be a
        # worse failure than the one being prevented. It says so in the record.
        return {"garments": 1, "ok": True, "extra": "",
                "why": f"probe failed: {type(e).__name__}"}
    return {"garments": n, "ok": n <= 1,
            "extra": str(v.get("extra", ""))[:200],
            "why": str(v.get("why", ""))[:200]}


def face_verdict(ref_face: dict, cand_face: dict) -> dict:
    """Is this candidate showing a different face from the source?

    Only a definite disagreement counts. CANNOT TELL on either side is not
    evidence of anything, and a 15-point penalty on a coin flip would be worse
    than not asking - the model reasons partly from what garments of this type
    usually look like, so its `why` is recorded with every flag and a flag is
    meant to be appealable.
    """
    out = {**cand_face, "flipped": False, "differs": ""}
    for key, label in (("side", "front/back"), ("face", "outside/inside")):
        a, b = ref_face.get(key), cand_face.get(key)
        if a in ("CANNOT TELL", None) or b in ("CANNOT TELL", None) or a == b:
            continue
        out["flipped"] = True
        out["differs"] = (f"{label}: the source reads {a}, this reads {b}"
                          + (f" - {cand_face.get('why','')}" if cand_face.get("why") else ""))
        break
    return out


def with_face(prompt: str, ask: bool = False) -> str:
    """Take the folded-question markers back out of the construction prompt.

    The markers stay in the template because the folded form is worth keeping
    documented as something that was tried, measured and rejected - see the
    comment above FACE_PROBE.
    """
    return prompt.replace("%FACE%\n", "").replace("%FACEKEY%", "")


def with_expected(prompt: str, expected: str) -> str:
    """Declare what the clean step legitimately removed, so its absence is not
    reported as a defect.

    Nothing else in the pipeline can tell the judge this. The cleaned source is
    what stage 3 compares against, so anything the clean was SUPPOSED to remove
    but did not - a pin it missed, a tag the guard refused to erase - is present
    in the reference, absent from every candidate, and reported as a
    discrepancy on all of them. That is what happened on
    runs/20260819_205617: three pearl-headed pins survived the clean, all ten
    candidates correctly left them out, and all thirty region verdicts came back
    MISMATCH. Ten paid-for images and thirty vision calls, none of which said
    anything about the garment.
    """
    if not expected:
        return prompt.replace("%EXPECTED%\n", "")
    return prompt.replace("%EXPECTED%",
                          f"\nAlso EXPECTED and NOT a defect - the clean-up step removed these\n"
                          f"before the candidate was made, so their absence is CORRECT:\n"
                          f"  {expected}\n")


PAIR_PROMPT = """Image 1 is the REFERENCE: the real garment, photographed flat as-is.
Images 2 and 3 are two GENERATED ecommerce flats of that same garment, competing
against each other. Call them CANDIDATE 1 (image 2) and CANDIDATE 2 (image 3).

Both are supposed to differ from the reference by having a cleaner background,
softer shadows, a straightened and re-centred garment, and relaxed creases. Those
are the goal, not defects. Judge the two candidates only against each other.

You MUST choose on each axis. "tie" is only for a genuinely indistinguishable pair,
and picking it for two visibly different images is a failure to do the task.

SHAPE - which is the better product silhouette? Look at left/right symmetry, whether
the legs are straight and evenly spread, waistband evenness, and any warping,
pinching or bowing.

WRINKLES - which surface is cleaner? Look for creases, fold shadows, blotchy or
uneven shading across large flat panels.

Return ONE JSON object, nothing else:
{{"shape_better": "1" | "2" | "tie",
  "shape_why": "<the deciding difference, one short sentence>",
  "wrinkles_better": "1" | "2" | "tie",
  "wrinkles_why": "<the deciding difference, one short sentence>"}}"""


def compare_pair(client: Client, ref_small: Path,
                 a_small: Path, b_small: Path) -> dict:
    content = [text_part("Image 1 - REFERENCE:"), image_part(ref_small),
               text_part("Image 2 - CANDIDATE 1:"), image_part(a_small),
               text_part("Image 3 - CANDIDATE 2:"), image_part(b_small),
               text_part(PAIR_PROMPT)]
    return parse_json_blob(client.chat(content, max_tokens=350, temperature=0.0))


def run_tournament(client: Client, ref_small: Path, rows: list[dict],
                   concurrency: int) -> dict:
    """Round-robin, every pair judged in BOTH orders.

    Absolute 0-100 rubric scoring saturates: this model returned exactly 100 on
    every axis for every candidate across 12 calls, including one with a visibly
    grey backdrop. Relative judgement still discriminates, and running each pair
    both ways lets position bias be measured rather than assumed away."""
    names = [r["name"] for r in rows]
    by_name = {r["name"]: r for r in rows}
    jobs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            jobs.append((names[i], names[j]))
            jobs.append((names[j], names[i]))     # same pair, swapped order

    wins = {n: {"shape": 0.0, "wrinkles": 0.0} for n in names}
    played = {n: 0 for n in names}
    pos_pick = {"1": 0, "2": 0, "tie": 0}
    reasons = []

    def one(a: str, b: str):
        return a, b, compare_pair(client, ref_small,
                                  by_name[a]["_small"], by_name[b]["_small"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, a, b) for a, b in jobs]
        for fut in concurrent.futures.as_completed(futs):
            try:
                a, b, v = fut.result()
            except Exception:  # noqa: BLE001 - a dropped bout is survivable
                continue
            played[a] += 1
            played[b] += 1
            for axis, key in (("shape", "shape_better"), ("wrinkles", "wrinkles_better")):
                pick = str(v.get(key, "tie")).strip().lower()
                if axis == "shape":
                    pos_pick[pick if pick in pos_pick else "tie"] += 1
                if pick == "1":
                    wins[a][axis] += 1
                elif pick == "2":
                    wins[b][axis] += 1
                else:
                    wins[a][axis] += 0.5
                    wins[b][axis] += 0.5
            reasons.append({"a": a, "b": b, **v})

    for n in names:
        g = max(played[n], 1)
        by_name[n]["shape"] = round(100.0 * wins[n]["shape"] / g, 1)
        by_name[n]["wrinkles"] = round(100.0 * wins[n]["wrinkles"] / g, 1)
        by_name[n]["bouts"] = played[n]
    return {"position_picks": pos_pick, "bouts": reasons}


def grade_once(client: Client, ref_small: Path, cand_small: Path,
               expected: str = "") -> dict:
    content = [text_part("Image 1 - REFERENCE:"), image_part(ref_small),
               text_part("Image 2 - CANDIDATE:"), image_part(cand_small),
               text_part(with_expected(GRADE_PROMPT, expected))]
    return parse_json_blob(client.chat(content, max_tokens=400, temperature=0.0))


def grade_voted(client: Client, ref_small: Path, cand_small: Path, votes: int,
                expected: str = "") -> dict:
    """Median of N independent grades. A single vision score is noisy evidence,
    not a measurement, so one call is never enough to rank on."""
    got = []
    for _ in range(votes):
        try:
            got.append(grade_once(client, ref_small, cand_small, expected))
        except Exception:  # noqa: BLE001 - a dropped vote is survivable
            continue
    if not got:
        raise RuntimeError("every grading vote failed")

    def med(key: str) -> float:
        vals = [float(g[key]) for g in got if isinstance(g.get(key), (int, float))]
        return round(statistics.median(vals), 1) if vals else 0.0

    verdicts = [str(g.get("verdict", "")).lower() for g in got]
    return {
        "shape": med("shape"),
        "wrinkles": med("wrinkles"),
        "background": med("background"),
        "shape_spread": round(max([float(g["shape"]) for g in got]) -
                              min([float(g["shape"]) for g in got]), 1),
        "wrinkle_spread": round(max([float(g["wrinkles"]) for g in got]) -
                                min([float(g["wrinkles"]) for g in got]), 1),
        "verdict": max(set(verdicts), key=verdicts.count) if verdicts else "unknown",
        "shape_issues": got[0].get("shape_issues", ""),
        "wrinkle_issues": got[0].get("wrinkle_issues", ""),
        "votes_counted": len(got),
    }


# --------------------------------------------------------------------------
# Stage 3 - construction integrity on native-resolution crops
# --------------------------------------------------------------------------

REGIONS = {           # name -> (top, bottom) as a fraction of the garment bbox
    "waistband": (0.00, 0.22),
    "hip/pocket": (0.18, 0.45),
    "hem": (0.80, 1.00),
}

# Bras are wider than tall and have no waistband or hem, so the leggings bands
# land on empty plate. profiles/ already splits the two categories, and
# C.garment_profile() picks between them from the run's own garment_type -
# --profile only overrides that.
REGIONS_BY_PROFILE = {
    "leggings": REGIONS,
    # The leggings bands, renamed for what a woven bottom carries at each
    # height. Same numbers on purpose - see PROFILE_TERMS in common.py for the
    # measurement that says the geometry holds. The names are not cosmetic: they
    # are what stage 3 tells the model to look at, and "waistband" alone does
    # not ask about a fly.
    "loose": {"waistband/fly": (0.00, 0.22),
              "pockets/rise": (0.18, 0.45),
              "hems": (0.80, 1.00)},
    # Same bands as loose - see PROFILE_TERMS in common.py for the check that
    # says the geometry holds on a gathered waist too. The top band is renamed
    # because what stage 3 has to catch there is different: the gathering
    # itself, which a generation flattens into a plain band, and the frill,
    # which it drops entirely.
    "boyfriend": {"waistband/gathers/frill": (0.00, 0.22),
                  "pockets/rise": (0.18, 0.45),
                  "hems": (0.80, 1.00)},
    "bras": {"band": (0.62, 1.00), "cups/centre": (0.25, 0.68),
             "straps": (0.00, 0.30)},
    # A pullover's bands are vertical spans at FULL garment width (crop_region
    # takes the whole bbox width), so each one carries the body AND the sleeve
    # at that height - which is what you want here: sleeve seams, armhole and
    # cuff ribbing are exactly the construction stage 3 exists to catch. The
    # narrow body-only boxes live in crop_pair.py, for looking at one thing.
    "pullovers": {"collar/shoulders": (0.00, 0.28),
                  "chest/sleeves": (0.22, 0.62),
                  "hem/cuffs": (0.76, 1.00)},
    # The pullover's bands, renamed for the placket. Every band names it because
    # it runs the whole height, and it is what stage 3 has to catch here: a
    # generation closes the front into a pullover, drifts the button spacing, or
    # loses the buttonhole side. "chest/sleeves" never asks about any of that.
    "cardigans": {"neckline/placket top": (0.00, 0.28),
                  "placket/buttons/sleeves": (0.22, 0.62),
                  "placket foot/hem/cuffs": (0.76, 1.00)},
    # Fleece gets the pullover's three bands moved to where its hardware is.
    # The top band runs deeper because a stand collar or hood is taller than a
    # crewneck and its seam would otherwise fall on the join between bands,
    # visible in neither. The bottom band starts at 0.70 rather than 0.76 so the
    # hand pockets sit inside it: their openings are the second thing after the
    # zip that a generation invents, and at 0.76 they land above the crop.
    #
    # These stay full-width like the others, so each band carries the sleeve at
    # that height as well as the body - and on a fleece the pile direction is
    # part of what has to match, which a full-width band shows and a body-only
    # box does not. The narrow boxes are in crop_pair.py, for looking at one
    # thing once this has told you which band moved.
    "fleeces": {"collar/hood/shoulders": (0.00, 0.30),
                "chest/zip/sleeves": (0.24, 0.62),
                "pockets/hem/cuffs": (0.70, 1.00)},
}


def crop_region(src: Path, box: tuple[int, int, int, int],
                span: tuple[float, float], tag: str) -> Path:
    """Crop a band of the garment at native resolution. Judging a whole 2K frame
    downscaled into the model loses exactly the fine detail this check exists to
    find.

    `box` is the garment's own bounding box in this image's pixels, and it is
    passed in rather than measured here so that both sides of a comparison are
    taken from boxes that were checked against each other. A band is a FRACTION
    of that box: get the box wrong by 10x, as the old luminance mask did on
    three candidates of runs/20260819_205617, and 'straps' lands on the band
    while the judge answers confidently about the wrong fabric.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bh <= 0 or bw <= 0:
        raise RuntimeError(f"no garment found in {src.name}")
    x, y = int(x0), int(y0 + span[0] * bh)
    w, h = int(bw), int((span[1] - span[0]) * bh)

    out = WORK / f"{src.stem}_{tag}.jpg"
    subprocess.run(["magick", str(src), "-crop", f"{w}x{h}+{x}+{y}", "+repage",
                    "-resize", "1024x1024>", str(out)],
                   check=True, capture_output=True)
    return out


def region_spec(regions: dict) -> str:
    """The crop bands themselves, as a string, for the cache fingerprint. Edit a
    band and every verdict taken with the old one stops matching, which is the
    behaviour you want: those verdicts were about a different piece of fabric."""
    return "|".join(f"{n}:{a:.3f}-{b:.3f}" for n, (a, b) in regions.items())


def fingerprint(cand: Path, ref: Path, profile: str, regions: dict,
                expected: str = "") -> dict:
    """What a stored verdict has to agree with before it can be reused.

    Content hashes, not filenames or mtimes: cand_02.png regenerated under the
    same name is a different image and must be re-judged, while the same bytes
    re-graded on a later pass are not.

    --expected-changes is part of it because it changes the question. A verdict
    taken before "the pins were removed on purpose" was declared answers a
    different question from one taken after, and reusing the first would leave
    the flag it exists to clear sitting in the record.
    """
    return {"cand_md5": C.md5(cand), "ref_md5": C.md5(ref),
            "profile": profile, "regions": region_spec(regions),
            "expected_changes": expected,
            # Bumped whenever the questions change. A verdict taken before the
            # orientation question existed does not carry an answer to it, and
            # reusing one would silently mark a flipped candidate as clean.
            "stage3": STAGE3_VERSION}


def load_cache(arch: Path) -> dict:
    """Stage-3 verdicts an earlier pass already paid for, keyed by candidate.

    They live in metrics.json so there is one record rather than a file that
    can disagree with it. A cache that cannot be read is not an error - it just
    means everything is judged fresh.
    """
    p = arch / "metrics.json"
    if not p.exists():
        return {}
    try:
        got = json.loads(p.read_text()).get("construction_cache") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return got if isinstance(got, dict) else {}


def check_construction(client: Client, ref: Path, cand: Path, regions: dict,
                       ref_box, cand_box, expected: str = "") -> list[dict]:
    out = []
    for i, (name, span) in enumerate(regions.items()):
        tag = f"r{i}"
        try:
            rc = crop_region(ref, ref_box, span, tag)
            cc = crop_region(cand, cand_box, span, tag)
        except Exception as e:  # noqa: BLE001
            out.append({"verdict": "ERROR", "detail": str(e)[:120], "region": name})
            continue
        content = [text_part(f"Image 1 - REFERENCE {name}:"), image_part(rc),
                   text_part(f"Image 2 - GENERATED {name}:"), image_part(cc),
                   # Formatted first, then the expected-changes text is pasted
                   # in: it is operator input and may contain braces of its own,
                   # which .format() would try to read as a field.
                   text_part(with_face(
                       with_expected(CONSTRUCTION_PROMPT.format(region=name),
                                     expected)))]
        try:
            v = parse_json_blob(client.chat(content, max_tokens=250,
                                            temperature=0.0))
        except Exception as e:  # noqa: BLE001
            v = {"verdict": "ERROR", "detail": str(e)[:120]}
        v["region"] = name
        out.append(v)
    return out


def flipped(r: dict) -> dict | None:
    """This candidate's face verdict, if it says the wrong face is showing."""
    f = r.get("face") or {}
    return f if f.get("flipped") else None


# --------------------------------------------------------------------------

def normalise(vals: list[float], lower_is_better: bool) -> list[float]:
    """Map a raw metric onto 0-100 within this batch.

    Batch-relative by construction, so 100 means "best of these" and nothing
    more. It is kept for the two starred context columns and is deliberately no
    longer part of the grade: as the headline wrinkle term it handed 100/100 to
    whichever candidate had ironed the fabric flattest, which on
    runs/20260819_205617 was a redraw with three altered regions.
    """
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [100.0] * len(vals)
    out = [(v - lo) / (hi - lo) for v in vals]
    return [round((1 - o if lower_is_better else o) * 100, 1) for o in out]


def band(value: float, best: float, worst: float) -> float:
    """A measurement onto 0-100 between two ANCHORS, not between its neighbours.

    Anchored scoring is the point: it survives a batch where everything is bad,
    which batch-relative scoring cannot - four ranked images look identical
    whether they are four good ones or four terrible ones.
    """
    span = best - worst
    if abs(span) < 1e-9:
        return 100.0
    return max(0.0, min(100.0, (value - worst) / span * 100))


def score_row(m: dict, args, ref_wrinkle: float) -> dict:
    """The five measured terms of one candidate, each 0-100.

    Anchors, and where they come from - all measured on runs/20260819_205617,
    ten 2K candidates of a mint bralette, the only batch this has been
    calibrated against:

      silhouette  IoU 0.95 scores 100, 0.70 scores 0. A re-lay cannot score 1.0
                  and should not: closing the straps is the job. The faithful
                  candidates measured 0.85-0.88 there and the redrawn ones
                  0.77-0.81.
      colour      dE76 1.0 or under scores 100 (invisible), 6.0 scores 0 (a
                  different colourway). Faithful 1.5-3.0, redrawn 3.8-5.3.
      wrinkle     distance from the source's own texture energy, tolerated up
                  to --wrinkle-tol of it in EITHER direction. Rougher means
                  creases the re-lay failed to relax; smoother means the knit
                  was ironed out of existence, which is the failure the old
                  batch-relative term rewarded.
      symmetry    mirror IoU of the silhouette against itself, 1.00 to
                  --sym-floor. Presentation, not fidelity - a redraw is usually
                  MORE symmetric than the real thing, which is why it is 15%.
      background  the plate this pipeline actually produces, which sweeps
                  228-252 and never reaches pure white.
    """
    tol = max(args.wrinkle_tol * ref_wrinkle, 1e-6)
    return {
        "silhouette": round(band(m.get("silhouette_iou", 0.0),
                                 args.iou_ceiling, args.iou_floor), 1),
        "colour": round(band(m.get("colour_de", 99.0), args.de_free, args.de_max), 1),
        "wrinkle": round(band(abs(m.get("wrinkle_delta", tol)), 0.0, tol), 1),
        "symmetry": round(band(m.get("symmetry", 0.0), 1.0, args.sym_floor), 1),
        "background": round(band(m.get("bg_lum", 0.0), args.bg_white, args.bg_floor), 1),
    }


def build_sheet(ref_small: Path, rows: list[dict], out: Path) -> Path:
    tiles = WORK / "tiles"
    if tiles.exists():
        shutil.rmtree(tiles)
    tiles.mkdir(parents=True)
    subprocess.run(["magick", str(ref_small), "-resize", "420x560", "-gravity", "north",
                    "-background", "white", "-splice", "0x46", "-font", FONT,
                    "-pointsize", "28", "-fill", "red", "-annotate", "+0+8",
                    "REFERENCE", str(tiles / "00.jpg")], check=True, capture_output=True)
    for i, r in enumerate(rows, start=1):
        colour = {"PASS": "blue", "REJECT": "red"}.get(r["status"], "gray50")
        label = f'{r["name"]}  {r["grade"]:.0f}  {r["status"]}'
        subprocess.run(["magick", str(r["_small"]), "-resize", "420x560",
                        "-gravity", "north", "-background", "white", "-splice", "0x46",
                        "-font", FONT, "-pointsize", "26", "-fill", colour,
                        "-annotate", "+0+8", label, str(tiles / f"{i:02d}.jpg")],
                       check=True, capture_output=True)
    subprocess.run(["montage", *sorted(str(p) for p in tiles.glob("*.jpg")),
                    "-font", FONT, "-tile", f"{len(rows) + 1}x", "-geometry", "+6+6",
                    "-background", "gray90", str(out)], check=True, capture_output=True)
    return out


def write_metrics_json(arch: Path, rows: list[dict], ref: Path, args,
                       profile: str, profile_resolved: bool, cache: dict) -> Path:
    """Write archive/metrics.json - the run's machine-readable verdict.

    Nothing consumes this automatically any more; review.py, which used to
    cross-check picks against it, is no longer part of the pipeline. It stays
    because it is the only per-candidate record with the construction verdicts
    in it, and `## Results` in the log has to be written from measured numbers
    rather than remembered ones.

    It is also where stage-3 verdicts are stored between passes, under
    `construction_cache`. Written whole every time, including entries for
    candidates this pass did not judge, so a `--no-construction` or
    `--candidates`-narrowed run cannot quietly delete a verdict that was paid
    for earlier.
    """
    out = {
        "schema": "grade",
        "reference": str(ref),
        "min_grade": args.min_grade,
        "judge": args.judge,
        "weights": WEIGHTS,
        "construction_penalty": args.construction_penalty,
        "expected_changes": args.expected_changes,
        # Which regions the construction verdicts below were taken on, and
        # whether that was read from the run or assumed. A verdict read without
        # this cannot be checked afterwards.
        "profile": profile,
        "profile_resolved": profile_resolved,
        "candidates": [{
            "cand": r["name"],
            "file": str(r["path"]),
            "score": r["grade"],
            "grade": r["grade"],
            "grade_before_penalty": r["base"],
            "penalty": r["penalty"],
            "status": r["status"],
            "reject": r["status"] != "PASS",
            "reject_why": "; ".join(r["notes"]) if r["status"] != "PASS" else "",
            # The five measured terms, each 0-100, in the weights above.
            "terms": r["terms"],
            # ... and the raw measurements they were scored from, so a number in
            # `## Results` can be checked rather than taken on trust.
            "silhouette_iou": r["metrics"].get("silhouette_iou"),
            "colour_de": r["metrics"].get("colour_de"),
            "colour_drgb": r["metrics"].get("colour_drgb"),
            "wrinkle_energy": r["metrics"].get("wrinkle_energy"),
            "wrinkle_delta": r["metrics"].get("wrinkle_delta"),
            "symmetry": r["metrics"].get("symmetry"),
            "asymmetry": r["metrics"].get("asymmetry"),
            "bg_lum": r["metrics"].get("bg_lum"),
            "rgb": r["metrics"].get("rgb"),
            "mask_cue": r["metrics"].get("cue"),
            "vision": r.get("vision"),
            "flipped": r.get("flipped", False),
            "face": r.get("face", {}),
            "construction": r.get("construction", []),
            # Judged this pass, or reused from a previous one. `## Results` can
            # then say when the flags it quotes were actually decided.
            "construction_from": r.get("construction_from", "not checked"),
        } for r in rows],
        "construction_cache": cache,
    }
    p = arch / "metrics.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="the run folder prepare.py printed as RUN_DIR=")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reference", type=Path, default=None,
                    help="default <run>/archive/offset_upload.jpg - the CLEANED "
                         "image the generator actually received. Do not point "
                         "this at inputs/off_set_image.jpg: the raw input still "
                         "has the hang tag, and stage 3 then reports the "
                         "correctly-removed tag as a MISMATCH on everything.")
    ap.add_argument("--candidates", type=Path, default=None,
                    help="default <run>/archive, auto-selecting every generated "
                         "image at the highest resolution present.")
    ap.add_argument("--profile", choices=sorted(REGIONS_BY_PROFILE), default=None,
                    help="which crop regions stage 3 uses. Read automatically "
                         "from the garment_type in <run>/reference_selection.json "
                         "- pass this only to override that. Bras have no "
                         "waistband or hem, so the leggings bands land on plate.")
    ap.add_argument("--judge", choices=["metrics", "tournament", "absolute"],
                    default="metrics",
                    help="whether to ALSO ask the vision model, and how. The grade "
                         "itself is always the measurements: 'tournament' showed "
                         "100%% position bias here and 'absolute' saturated at "
                         "100/100 for every candidate including one with a visibly "
                         "grey backdrop, so neither is allowed near the score. "
                         "They print beside it, for re-testing on a better model.")
    ap.add_argument("--votes", type=int, default=3,
                    help="independent grading calls per candidate, median taken "
                         "(default 3; vision scores are too noisy for one)")
    ap.add_argument("--min-grade", "--threshold", type=float, default=62.0,
                    metavar="GRADE",
                    help="below this a candidate is not shippable. Default 62 is "
                         "anchored on the only calibrated batch: the five "
                         "candidates a human ranked as faithful re-lays scored "
                         "63.8-74.3 and the three redrawn ones topped out at "
                         "47.5. It is a pass mark for FIDELITY, so it is not "
                         "comparable to the 80 the old presentation grade used.")
    ap.add_argument("--iou-floor", type=float, default=0.70,
                    help="silhouette IoU against the source scoring 0 (default "
                         "0.70 - below that it is not the same outline)")
    ap.add_argument("--iou-ceiling", type=float, default=0.95,
                    help="silhouette IoU scoring 100 (default 0.95; 1.0 would "
                         "mean the garment was never re-laid at all)")
    ap.add_argument("--de-free", type=float, default=1.0,
                    help="colour difference (dE76) still scoring 100 - below "
                         "this it is invisible (default 1.0)")
    ap.add_argument("--de-max", type=float, default=6.0,
                    help="colour difference scoring 0 (default 6.0, a different "
                         "colourway)")
    ap.add_argument("--wrinkle-tol", type=float, default=0.35,
                    help="how far a candidate's texture energy may sit from the "
                         "SOURCE's own, as a fraction of it, before the term "
                         "scores 0. Both directions: smoother than the real "
                         "garment is a redraw, not a win (default 0.35)")
    ap.add_argument("--sym-floor", type=float, default=0.80,
                    help="mirror IoU scoring 0 for symmetry (default 0.80)")
    ap.add_argument("--construction-penalty", type=float, default=15.0,
                    metavar="POINTS",
                    help="points off the grade per region stage 3 flagged as "
                         "altered (default 15). Without it the grade and the "
                         "MISMATCH label contradict each other and someone has "
                         "to arbitrate; one run spent thirty turns doing that.")
    ap.add_argument("--expected-changes", default="", metavar="TEXT",
                    help="what the clean step legitimately removed, declared to "
                         "the judges so their absence is not reported as a "
                         "defect: --expected-changes \"pearl-headed pins "
                         "removed\". Stored verdicts taken without it are "
                         "re-judged, because it changes the question.")
    # The imported defaults were 0.95 -> 0 and 1.00 -> 100, which on this
    # project's own output flagged "backdrop not white" on 10 candidates out of
    # 10 and zeroed the term for 6 of them. common.py records why: the plate
    # here is not white, it sweeps 228-252 (0.894-0.988), so a real clean plate
    # never reaches 1.00. A warning that fires on every candidate carries no
    # information, so the scale is anchored on the plate this pipeline actually
    # produces.
    ap.add_argument("--bg-floor", type=float, default=0.90,
                    help="backdrop lightness scoring 0 (default 0.90)")
    ap.add_argument("--bg-white", type=float, default=0.99,
                    help="backdrop lightness scoring 100 (default 0.99)")
    ap.add_argument("--no-construction", action="store_true",
                    help="skip the crop-level construction integrity check. "
                         "Stored verdicts are left alone, not discarded.")
    ap.add_argument("--rejudge", action="store_true",
                    help="re-run stage 3 on candidates that already have a "
                         "stored verdict, replacing it. Without this a verdict "
                         "is judged once per (candidate, reference, profile) "
                         "and reused, so grading twice cannot produce two "
                         "different sets of flags. This is the deliberate way "
                         "to get a second opinion - and it overwrites, so there "
                         "is still exactly one record.")
    ap.add_argument("--ship", type=int, default=0, metavar="N",
                    help="copy the top N candidates BY GRADE to <run>/output/, "
                         "regardless of status. The grade now carries a "
                         "construction penalty, so a flagged pick has to be "
                         "clearly better on everything else to outrank a clean "
                         "one - but it still can, and every MISMATCH shipped is "
                         "printed and logged.")
    ap.add_argument("--ship-faithful", type=int, default=0, metavar="N",
                    help="ship N, preferring candidates stage 3 found intact: "
                         "top N by grade among the unflagged, and only if fewer "
                         "than N of those exist does it backfill from the "
                         "flagged ones, printing exactly what it took and why. "
                         "This is the one that always delivers something AND "
                         "always says what it cost - use it instead of --ship "
                         "unless there is a reason not to.")
    ap.add_argument("--ship-clean-only", action="store_true",
                    help="restore the gate: ship only PASS candidates, so a "
                         "construction MISMATCH never reaches output/. Ships "
                         "fewer than asked, or nothing, rather than backfill.")
    ap.add_argument("--cutout", action="store_true",
                    help="also write a transparent-background *_cutout.png "
                         "beside each pick. Off by default: --ship currently "
                         "delivers the flats themselves, so the cutout step is "
                         "opt-in until it is wanted again.")
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    # One number and one mode from here on, so nothing downstream has to ask
    # which of the three flags was passed.
    if args.ship and args.ship_faithful:
        print("--ship and --ship-faithful both given; they select different "
              "picks. Choose one.", file=sys.stderr)
        return 1
    args.deliver = args.ship or args.ship_faithful
    args.ship_mode = ("faithful" if args.ship_faithful else
                      "clean" if args.ship_clean_only else "grade")

    sys.stdout.reconfigure(line_buffering=True)

    arch = args.run / "archive"
    ref = args.reference or (arch / "offset_upload.jpg")
    if not ref.exists():
        print(f"reference not found: {ref}\n"
              f"  Run prepare.py first - it writes the cleaned upload there.",
              file=sys.stderr)
        return 1

    if args.candidates and args.candidates.is_dir():
        cands = sorted(p for p in args.candidates.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        label = str(args.candidates)
    else:
        cands, label = auto_select(arch)
    if not cands:
        print(f"No candidates in {arch}. Run generate.py first.", file=sys.stderr)
        return 1

    # Stage 3 is the only check that can tell a re-laid garment from a redrawn
    # one, and it is worthless if its crops land on the wrong part of the
    # garment. So the profile is read off the run rather than left to a flag.
    profile, prof_why, prof_ok = C.garment_profile(args.run, args.profile)
    if profile not in REGIONS_BY_PROFILE:
        print(f"unknown profile {profile!r}; choose from "
              f"{', '.join(sorted(REGIONS_BY_PROFILE))}", file=sys.stderr)
        return 1

    # An assumed profile means stage 3 cropped somewhere it was never told to
    # look, so its verdicts say nothing about this garment. That used to be a
    # printed warning, which is to say it was ignored and a batch shipped on
    # verdicts taken from empty plate. Delivery is now refused outright, before
    # anything is graded, so the cost of the mistake is one flag rather than a
    # whole run. --no-construction does NOT unblock it: skipping the check is a
    # decision to ship unchecked, and it is not one to make by accident while
    # trying to get past this message.
    if not prof_ok and args.deliver:
        print(f"\nREFUSING TO SHIP: {prof_why}\n"
              f"  --ship writes the deliverable, and stage 3 - the only check "
              f"that can tell a re-laid garment from a redrawn one - crops by "
              f"garment. On the wrong profile it reads empty plate and returns "
              f"confident verdicts about nothing.\n"
              f"  Fix it: --profile {' | --profile '.join(sorted(REGIONS_BY_PROFILE))}\n"
              f"  Or find out why step 0 did not record a garment_type - "
              f"{args.run / 'reference_selection.json'} is where it belongs.\n"
              f"  Grading without --ship still runs, so you can look first.",
              file=sys.stderr)
        return 1

    # The grade is entirely measured now, so a run that only wants the grade
    # does not need the vision server at all. Resolving the model is therefore
    # allowed to fail when nothing is going to ask it anything - grading a batch
    # on a laptop with the server down is a legitimate thing to do, and it used
    # to die on line one.
    needs_model = (not args.no_construction) or args.judge != "metrics"
    client = Client(args.base_url, args.model, args.timeout)
    try:
        model = client.resolve_model()
        print(f"model: {model}")
    except Exception as e:  # noqa: BLE001
        if needs_model:
            print(f"cannot reach the vision server at {args.base_url}: {e}\n"
                  f"  Stage 3 needs it. --no-construction grades without it, "
                  f"and says so in the output.", file=sys.stderr)
            return 1
        model = "(none - measurements only)"
        print(f"model: {model}")
    print(f"reference:  {ref}   (the cleaned upload, not the raw input)")
    print(f"candidates: {len(cands)}   votes/candidate: {args.votes}")
    if args.expected_changes:
        print(f"expected:   {args.expected_changes}   <- declared to the judges "
              f"as correct, not a defect")
    # Only reachable unresolved without --ship, which is refused above.
    prof_line = f"profile:    {prof_why}"
    if not prof_ok:
        prof_line += (f"\n            grading anyway on {profile} regions, but "
                      f"stage 3 is looking at a garment nobody confirmed. "
                      f"--ship is refused until --profile says which.")
    print(prof_line + "\n")

    # Verdicts an earlier pass already paid for. Kept across the whole run so
    # --no-construction rewrites metrics.json without dropping them.
    cache = load_cache(arch)
    if args.rejudge and cache:
        print(f"--rejudge: {len(cache)} stored verdict(s) will be replaced\n")

    # --- Stage 1 ---------------------------------------------------------
    print("stage 1: measurements against the cleaned source (no model)")
    t0 = time.time()
    ref_m, ref_mask = measure(ref)
    ref_box = C.garment_box(ref)["box"]
    if not ref_m["found"]:
        print(f"\nNO GARMENT FOUND IN THE REFERENCE ({ref}).\n"
              f"  Every fidelity term is measured against it, so there is "
              f"nothing to grade. Look at the image before anything else.",
              file=sys.stderr)
        return 1
    print(f"  SOURCE      mask {ref_m['cue']}  {ref_m['coverage']*100:.1f}% of "
          f"frame, aspect {ref_m['aspect']:.2f}   rgb {ref_m['rgb']}   "
          f"wrinkle {ref_m['wrinkle_energy']:.3f}   sym {ref_m['symmetry']:.3f}   "
          f"bg_lum {ref_m['bg_lum']:.3f}")
    print(f"  {'':12}{'cue':<10} {'IoU':>6} {'dE':>6} {'wrinkle':>8} "
          f"{'d-wr':>7} {'sym':>6} {'bg':>6}")
    rows = []
    for p in cands:
        m, mask = measure(p, ref_m, ref_mask)
        g = C.garment_box(p, like=ref)
        rows.append({"name": p.stem, "path": p, "metrics": m, "box": g["box"],
                     "box_ok": g["ok"],
                     "_small": ensure_small(p, args.max_dim)})
        print(f"  {p.stem:<12}{m['cue']:<10} {m.get('silhouette_iou', 0)*100:6.1f} "
              f"{m.get('colour_de', 0):6.2f} {m.get('wrinkle_energy', 0):8.3f} "
              f"{m.get('wrinkle_delta', 0):+7.3f} {m.get('symmetry', 0)*100:6.1f} "
              f"{m.get('bg_lum', 0)*100:6.1f}")
        if not g["ok"]:
            print(f"  {'':12}{g['note']}")
    print(f"  done in {time.time() - t0:.1f}s")

    # Two batch-relative context columns, printed but no longer scored. They
    # answer "which of these is smoothest" - a question the grade deliberately
    # stopped asking, because the answer is usually the redraw.
    sym_n = normalise([r["metrics"]["asymmetry"] for r in rows], True)
    wr_n = normalise([r["metrics"]["wrinkle_energy"] for r in rows], True)
    for r, s, w in zip(rows, sym_n, wr_n):
        r["sym_rank"], r["wr_rank"] = s, w

    # --- Stage 2, advisory -------------------------------------------------
    # Nothing below feeds the grade. Both model modes were measured on this
    # project and both failed in ways that cannot be corrected for: the
    # tournament picked by slot 100% of the time, and the rubric returned
    # 100/100 for every candidate across 12 calls including one with a visibly
    # grey backdrop. They are kept because re-testing them on a better model
    # should cost one flag, not a rewrite.
    ref_small = ensure_small(ref, args.max_dim)
    tour = {"position_picks": {}, "bouts": []}
    for r in rows:
        r["shape"] = r["wrinkles"] = 0.0
        r["bouts"] = 0

    if args.judge == "tournament":
        n_pairs = len(rows) * (len(rows) - 1)
        print(f"\nstage 2: pairwise tournament, {n_pairs} bouts "
              f"(every pair judged in both orders)")
        t0 = time.time()
        transient(f"  running {n_pairs} comparisons ...")
        tour = run_tournament(client, ref_small, rows, args.concurrency)
        settled(f"  {n_pairs} bouts in {time.time() - t0:.1f}s")
        for r in sorted(rows, key=lambda r: -(r["shape"] + r["wrinkles"])):
            print(f"  {r['name']:<10}  shape win-rate {r['shape']:5.1f}   "
                  f"wrinkles win-rate {r['wrinkles']:5.1f}   ({r['bouts']} bouts)")
        # Both orders are judged, so a healthy judge splits its picks roughly
        # evenly between slot 1 and slot 2. A lopsided split means it is reading
        # position, not the images.
        pp = tour["position_picks"]
        total = max(pp["1"] + pp["2"], 1)
        skew = abs(pp["1"] - pp["2"]) / total
        print(f"  position check: slot-1 picks {pp['1']}, slot-2 picks {pp['2']}, "
              f"ties {pp['tie']}  -> skew {skew:.0%}")
        if skew > 0.4:
            print("  WARNING: the judge is picking by slot, not by image. These "
                  "win-rates are noise; prefer --judge metrics.")
    elif args.judge == "absolute":
        print(f"\nstage 2: absolute rubric grading, {args.votes} votes each "
              f"(advisory - does not feed the grade)")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(grade_voted, client, ref_small, r["_small"],
                                args.votes, args.expected_changes): r for r in rows}
            for fut in concurrent.futures.as_completed(futs):
                r = futs[fut]
                try:
                    g = fut.result()
                except Exception as e:  # noqa: BLE001
                    g = {"shape": 0.0, "wrinkles": 0.0, "shape_spread": 0.0,
                         "wrinkle_spread": 0.0, "verdict": f"error: {e}"[:60],
                         "votes_counted": 0}
                r["shape"], r["wrinkles"] = g["shape"], g["wrinkles"]
                r["bouts"] = g["votes_counted"]
                r["grade_detail"] = g
                r["vision"] = {k: g[k] for k in ("shape", "wrinkles", "verdict")}
                print(f"  {r['name']:<10}  shape {g['shape']:5.1f} "
                      f"(spread {g['shape_spread']:4.1f})   wrinkles {g['wrinkles']:5.1f} "
                      f"(spread {g['wrinkle_spread']:4.1f})   {g['verdict']}")
        print(f"  done in {time.time() - t0:.1f}s")
        if all(r["shape"] >= 99 and r["wrinkles"] >= 99 for r in rows):
            print("  WARNING: every candidate scored ~100 on both axes. The rubric "
                  "has saturated and cannot rank; prefer --judge metrics.")
    else:
        print("\nstage 2: skipped - the grade is measured, and no vision judge "
              "here beat the measurements. --judge absolute | tournament runs "
              "one anyway, beside the grade.")

    # --- Stage 3 ---------------------------------------------------------
    regions = REGIONS_BY_PROFILE[profile]
    n_reused = 0
    ref_face = {}
    if not args.no_construction:
        print(f"\nstage 3: construction integrity on native-res crops, "
              f"{profile} regions ({', '.join(regions)})")
        ref_face = face_of(client, ref_small)
        print(f"  orientation: the source reads {ref_face['side']} / "
              f"{ref_face['face']} - {ref_face['why'][:80]}")
        if "CANNOT TELL" in (ref_face["side"], ref_face["face"]):
            print("  the source's own face could not be read, so no candidate "
                  "can be compared against it; orientation is not checked.")
        t0 = time.time()
        n_judged = 0
        if args.expected_changes:
            print(f"  declared as expected, not defects: {args.expected_changes}")
        for r in rows:
            fp = fingerprint(r["path"], ref, profile, regions,
                             args.expected_changes)
            hit = cache.get(r["name"]) or {}
            fresh = args.rejudge or any(hit.get(k) != v for k, v in fp.items())
            if fresh:
                transient(f"  {r['name']:<10}  judging ...")
                r["construction"] = check_construction(
                    client, ref, r["path"], regions, ref_box, r["box"],
                    args.expected_changes)
                r["face"] = face_verdict(ref_face, face_of(client, r["_small"]))
                r["count"] = count_garments(client, r["_small"])
                stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                r["construction_from"] = f"judged {stamp[11:]}"
                n_judged += 1
                # An ERROR is a call that did not happen, not a verdict about
                # the garment. Caching one would freeze a dropped connection
                # into the record and never ask again.
                if not any(c.get("verdict") == "ERROR" for c in r["construction"]):
                    cache[r["name"]] = {**fp, "judged_at": stamp, "model": model,
                                        "verdicts": r["construction"],
                                        "face": r["face"],
                                        "count": r["count"]}
                else:
                    cache.pop(r["name"], None)
            else:
                r["construction"] = hit["verdicts"]
                r["face"] = hit.get("face") or {}
                # Records predating the duplicate check have no count. Treat
                # that as unknown-but-passing rather than re-probing every
                # cached candidate.
                r["count"] = hit.get("count") or {"garments": 1, "ok": True}
                r["construction_from"] = f"cached {hit.get('judged_at', '')[11:]}"
                n_reused += 1

            note = (f"   ({r['construction_from']})" if not fresh else "")
            bad = [c for c in r["construction"] if c.get("verdict") == "MISMATCH"]
            flip = flipped(r)
            dup = not (r.get("count") or {}).get("ok", True)
            head = ", ".join(filter(None, [
                f"MISMATCH in {', '.join(c['region'] for c in bad)}" if bad else "",
                "FLIPPED - shows the other face" if flip else "",
                f"{r['count']['garments']} GARMENTS - not one" if dup else ""]))
            settled(f"  {r['name']:<10}  {head or 'all regions match'}{note}")
            for c in bad:
                print(f"                -> {c['region']}: {c.get('detail','')}")
            if flip:
                print(f"                -> face: {flip.get('differs','')}")
            if dup:
                print(f"                -> duplicate: "
                      f"{r['count'].get('extra') or r['count'].get('why','')}")
        # The same images judged twice give two different answers - this model
        # scored one candidate 100 then 60 on a rescale alone - so a second pass
        # over the same batch reuses the first pass's verdict instead of rolling
        # again. That is what makes the flags in output/, in the ranking and in
        # LOG.md the same flags rather than three independent samples.
        print(f"  done in {time.time() - t0:.1f}s   "
              f"{n_judged} judged, {n_reused} reused from metrics.json"
              + ("  (--rejudge: cache overwritten)" if args.rejudge and n_judged
                 else ""))
        if n_reused and not args.rejudge:
            print("  reused verdicts were not re-rolled. --rejudge forces a "
                  "fresh judgement and replaces the stored one.")

        # Independent generations fail in independent ways. When every one of
        # them is flagged in the same region, the common cause is far more
        # likely to be the crop than the candidates - the wrong profile puts
        # 'waistband' on empty plate, and the judge dutifully reports the whole
        # band as missing on all of them. Check the region before believing it.
        flagged = [{c["region"] for c in r["construction"]
                    if c.get("verdict") == "MISMATCH"} for r in rows]
        shared = set.intersection(*flagged) if len(rows) > 1 and all(flagged) else set()
        if shared:
            print(f"\n  WARNING: all {len(rows)} candidates are flagged in "
                  f"{', '.join(sorted(shared))}. Independent draws rarely fail "
                  f"identically, so check the crops before the candidates:")
            print(f"    profile in use: {prof_why}")
            print(f"    look at one pair - crop_pair.py --run {args.run} --cand NN "
                  f"--at {sorted(shared)[0].split('/')[0]}")
            if not prof_ok:
                print("    the profile was ASSUMED, not read. That is the first "
                      "thing to rule out.")

        n_dup = sum(1 for r in rows if not (r.get("count") or {}).get("ok", True))
        if n_dup:
            # Same reasoning as the flip note: several at once is the prompt,
            # not the dice. The clause that suppresses it is appended by
            # generate.py, so a run with duplicates is worth checking against
            # the sent prompt rather than re-rolling.
            print(f"\n  {n_dup} of {len(rows)} candidates contain MORE THAN ONE "
                  f"GARMENT. That is a prompt fault, not a draw-by-draw one: "
                  f"check that archive/prompt_*.txt still carries the "
                  f"'DRAW EXACTLY ONE GARMENT' clause, and that the source is "
                  f"not being sent with large empty margins, which gives the "
                  f"model somewhere to put a second copy.")
        n_flip = sum(1 for r in rows if flipped(r))
        if n_flip:
            print(f"\n  {n_flip} of {len(rows)} candidates show the OTHER FACE of "
                  f"the garment. That is a prompt fault, not a draw-by-draw one: "
                  f"check archive/prompt.txt for anything naming a side, and the "
                  f"lay reference for whether it shows the face the product does "
                  f"not.")
        for r in rows:
            r["construction"] = []

    # --- Combine ---------------------------------------------------------
    for r in rows:
        t = score_row(r["metrics"], args, ref_m["wrinkle_energy"])
        r["terms"] = t
        r["background"] = t["background"]
        base = sum(WEIGHTS[k] * t[k] for k in WEIGHTS)
        notes = []
        if r["metrics"]["bg_lum"] < args.bg_floor:
            notes.append(f"backdrop not a plate (bg_lum {r['metrics']['bg_lum']:.3f} "
                         f"< {args.bg_floor:.2f})")
        if not r["box_ok"]:
            notes.append("garment box not credible; measured on a fallback")
        mism = [c for c in r.get("construction", []) if c.get("verdict") == "MISMATCH"]
        flip = flipped(r)

        # Construction failure costs points AND disqualifies. The penalty is not
        # a softening of the gate: it exists so the ranking cannot say one thing
        # while the status says another, which is what sent a run into thirty
        # turns of arbitration between a 79.1 grade and three MISMATCH flags on
        # the same image.
        #
        # A flip costs the same as one altered region and disqualifies the same
        # way. It is arguably worse than any single seam - it is the wrong side
        # of the product - but it is one fault, and pricing it as one keeps the
        # arithmetic something a reader can check.
        # A duplicate garment is priced and disqualified exactly like a flip.
        # It is one fault and it is fatal: the deliverable is a photograph of a
        # product, and one showing two of them is not usable at any grade. It
        # needs its own term because nothing else sees it - every geometric
        # metric here runs on the largest connected blob, so the second garment
        # is discarded before it can move a number. That is why cand_04 of
        # runs/20260827_095355 ranked SECOND with a whole extra sweater in frame.
        dup = not (r.get("count") or {}).get("ok", True)
        r["duplicate"] = bool(dup)
        r["flipped"] = bool(flip)
        r["penalty"] = round(
            args.construction_penalty * (len(mism) + bool(flip) + bool(dup)), 1)
        grade = base - r["penalty"]
        if dup:
            notes.append(
                f"DUPLICATE GARMENT (-{args.construction_penalty:.0f}): "
                f"{r['count'].get('garments')} garments in frame - "
                f"{r['count'].get('extra') or r['count'].get('why', '')}")
        if flip:
            notes.append(f"FLIPPED (-{args.construction_penalty:.0f}): "
                         f"{flip.get('differs', '')}")
        if mism:
            notes.append(f"construction altered ({len(mism)} region(s), "
                         f"-{args.construction_penalty * len(mism):.0f}): " +
                         "; ".join(f"{c['region']} - {c.get('detail','')}" for c in mism))
        # One chain, so nothing can set a status and have a later branch quietly
        # set it back. A flip disqualifies exactly as an altered region does: it
        # is the wrong side of the product, whatever the grade says.
        if mism or flip or dup:
            r["status"] = "REJECT"
        elif grade >= args.min_grade:
            r["status"] = "PASS"
        else:
            r["status"] = "BELOW"
            notes.append(f"grade {grade:.1f} < {args.min_grade:.0f}")
        r["base"] = round(max(0.0, min(100.0, base)), 1)
        r["grade"] = round(max(0.0, min(100.0, grade)), 1)
        r["notes"] = notes

    rows.sort(key=lambda r: (r["status"] == "REJECT", -r["grade"]))

    formula = " + ".join(f"{int(w*100)}% {k}" for k, w in WEIGHTS.items())
    print(f"\nRANKING  (grade = {formula}, "
          f"-{args.construction_penalty:.0f}/altered region, "
          f"pass mark {args.min_grade:.0f})")
    print(f"  {'':3} {'grade':>6} {'status':<7} {'name':<10} {'silh':>6} {'col':>6} "
          f"{'wrink':>6} {'sym':>5} {'bg':>5} {'pen':>5} {'sym*':>5} {'smooth*':>7}"
          f"  notes")
    for i, r in enumerate(rows, start=1):
        t = r["terms"]
        print(f"  {i:>2}. {r['grade']:6.1f} {r['status']:<7} {r['name']:<10} "
              f"{t['silhouette']:6.1f} {t['colour']:6.1f} {t['wrinkle']:6.1f} "
              f"{t['symmetry']:5.0f} {t['background']:5.0f} "
              f"{-r['penalty'] if r['penalty'] else 0:5.0f} "
              f"{r['sym_rank']:5.0f} {r['wr_rank']:7.0f}  "
              f"{'; '.join(r['notes'])[:52]}")
    print("  silh / col / wrink are measured AGAINST THE SOURCE: outline IoU, "
          "colour dE, texture distance")
    print("  * sym / smooth are batch-relative context, not scored - 100 means "
          "'most of these', not 'good'")
    if args.judge != "metrics":
        print(f"  the {args.judge} judge ran but does not feed the grade; its "
              f"numbers are above, in stage 2")
    least = min(rows, key=lambda r: len([c for c in r.get("construction", [])
                                         if c.get("verdict") == "MISMATCH"]))
    n_bad = len([c for c in least.get("construction", []) if c.get("verdict") == "MISMATCH"])
    if any(r["status"] == "REJECT" for r in rows):
        print(f"\n  least-altered candidate: {least['name']} ({n_bad} region(s) "
              f"changed).  Rank #{rows.index(least) + 1} of {len(rows)}.")

    winners = [r for r in rows if r["status"] == "PASS"]
    sheet = build_sheet(ref_small, rows, arch / "grade_results.jpg")
    (arch / "grade_results.json").write_text(json.dumps({
        "model": model,
        "reference": str(ref),
        "reference_metrics": ref_m,
        "profile": profile,
        "profile_source": prof_why,
        "profile_resolved": prof_ok,
        "votes": args.votes,
        "min_grade": args.min_grade,
        "best": winners[0]["name"] if winners else None,
        "position_check": tour["position_picks"],
        "bouts": tour["bouts"],
        "candidates": [{k: v for k, v in r.items()
                        if k not in ("_small", "path")} for r in rows],
    }, indent=2, default=str))
    mp = write_metrics_json(arch, rows, ref, args, profile, prof_ok, cache)

    print()
    if winners:
        b = winners[0]
        t = b["terms"]
        print(f"BEST: {b['name']}  grade {b['grade']:.1f}  "
              f"(silhouette {t['silhouette']:.0f}, colour {t['colour']:.0f}, "
              f"wrinkle {t['wrinkle']:.0f})")
        print(f"KEEP  {','.join(r['name'] for r in winners)}"
              f"  <- only these are eligible for --ship")
    else:
        best = rows[0] if rows else None
        print(f"NO SHIPPABLE CANDIDATE - nothing cleared {args.min_grade:.0f} "
              f"with construction intact")
        if best is not None:
            print(f"  closest was {best['name']} at {best['grade']:.1f}"
                  + (f" ({best['base']:.1f} before -{best['penalty']:.0f} for "
                     f"construction)" if best["penalty"] else ""))
            if all(r["penalty"] for r in rows) and not args.expected_changes:
                print("  EVERY candidate lost points to construction flags. If "
                      "the clean step left something behind - a pin, a clip, a "
                      "tag - every candidate correctly dropping it reads as ten "
                      "identical defects. Declare it: "
                      "--expected-changes \"...\" (and see the WARNING above).")
    print(f"wrote {arch / 'grade_results.json'}")
    print(f"wrote {mp}   <- per-candidate verdicts, incl. construction")
    print(f"wrote {sheet}")
    C.log(args.run, f"graded {len(rows)} ({label}), {len(winners)} shippable"
                    + (f", {profile} regions" if not args.no_construction else
                       ", no construction check")
                    + (f" ({n_reused} verdict(s) reused)" if n_reused else ""))

    if args.deliver:
        return ship(args, winners, rows)
    return 0 if winners else 2


def mismatched(r: dict) -> list[dict]:
    return [c for c in r.get("construction", []) if c.get("verdict") == "MISMATCH"]


def choose_picks(args, winners: list[dict], rows: list[dict]) -> tuple[list[dict], str, list[str]]:
    """Which candidates ship, under whichever of the three modes was asked for.

    Returns (picks, basis, notes). The notes are the whole point of the
    `faithful` mode: a delivery that quietly swapped a flagged image in for a
    missing clean one looks exactly like a delivery that had four clean ones,
    and the difference is the entire content of `## Picking`.
    """
    by_grade = sorted(rows, key=lambda r: -r["grade"])
    n = args.deliver
    if args.ship_mode == "clean":
        return winners[:n], "PASS only", []
    if args.ship_mode == "grade":
        # rows is sorted with REJECTs last; re-sort on grade alone so status
        # plays no part in who ships.
        return by_grade[:n], "top by grade", []

    clean = [r for r in by_grade if not mismatched(r)]
    picks, notes = clean[:n], []
    if len(picks) < n:
        # Deliberate, and deliberately loud. A batch where the construction
        # check flagged everything is a real outcome - it happened on a real
        # run, ten for ten - and shipping nothing leaves the images paid for
        # and undelivered. Backfilling by grade at least ships the best of a
        # bad batch, with the cost of each one named on its own line.
        short = n - len(picks)
        backfill = [r for r in by_grade if mismatched(r)][:short]
        notes.append(f"only {len(picks)} of {n} candidates came through stage 3 "
                     f"intact; backfilling {len(backfill)} by grade")
        for r in backfill:
            notes.append(f"  {r['name']} carries {len(mismatched(r))} altered "
                         f"region(s): "
                         f"{', '.join(c['region'] for c in mismatched(r))}")
        picks = picks + backfill
    excluded = [r for r in by_grade if mismatched(r) and r not in picks]
    if excluded:
        notes.append("held back for altered construction, best first: "
                     + ", ".join(f"{r['name']} ({r['grade']:.1f}, "
                                 f"{len(mismatched(r))} region(s))"
                                 for r in excluded[:4]))
    return picks, "faithful first, then by grade", notes


def ship(args, winners: list[dict], rows: list[dict]) -> int:
    """Copy the picks to output/.

    Three modes, and which one a run uses is the whole of its delivery policy:

      --ship N            top N by grade, status ignored.
      --ship-faithful N   top N among the candidates stage 3 found intact,
                          backfilled by grade only if there are fewer than N,
                          with every exclusion and every backfill printed.
      --ship-clean-only   PASS only, ships fewer rather than backfill.

    `faithful` exists because the other two are the two ways a run goes wrong.
    Ranking by grade alone shipped a redrawn garment at the top of a real
    batch; gating on PASS left output/ empty on another and the run with
    nothing to show for ten paid-for images. Neither needed a judgement call -
    they needed a rule that prefers the intact ones and says out loud when it
    could not find enough.

    What that costs, so it is not rediscovered later: the grade now carries
    -15 per altered region, so the ranking no longer contradicts the flags the
    way it did when the top-graded image of a real batch had three altered
    regions and the least-altered one ranked sixth. It is still a penalty and
    not a veto: a heavily flagged candidate that is otherwise excellent can
    outrank a clean mediocre one. Every MISMATCH is therefore printed per pick
    and written to steps.log, so a shipped defect is recorded rather than
    merely allowed.

    --ship-clean-only restores the gate.

    The picks are copied byte for byte. Only the cutout is a new file, and
    nothing is ever resampled on the way out.

    Cutouts are off unless --cutout is passed, so what lands in output/ is the
    generated flat on its own white plate. The retouch team then has no alpha
    channel to place against, which is a change to what the run delivers rather
    than a change to how it is produced - worth saying out loud in the log.
    """
    outd = args.run / "output"
    arch = args.run / "archive"

    picks, basis, why = choose_picks(args, winners, rows)
    flag = "--ship-faithful" if args.ship_mode == "faithful" else "--ship"
    if not picks:
        print(f"\n{flag} {args.deliver}: nothing to ship "
              f"({'the KEEP list is empty' if args.ship_mode == 'clean' else 'no candidates'}).")
        return 2

    outd.mkdir(parents=True, exist_ok=True)
    # Clear previous picks. Without this a second run leaves the first set
    # behind under different names and output/ no longer says what shipped.
    # This also sweeps cutouts from an earlier --cutout run, which would
    # otherwise sit in output/ looking like part of this delivery.
    for old in sorted(outd.glob("pick*.png")):      # includes _cutout.png
        old.unlink()

    print(f"\n{flag} {args.deliver}: writing {len(picks)} pick(s) to {outd}  "
          f"({basis})" + ("" if args.cutout else "  (flats only, no cutouts)"))
    for line in why:
        print(f"  {line}")
    shipped_bad = []
    for rank, r in enumerate(picks, 1):
        src = arch / f"{r['name']}.png"
        if not src.exists():
            print(f"  MISSING {src} - skipped")
            continue
        dst = outd / (f"pick{rank}_best_{r['name']}.png" if rank == 1
                      else f"pick{rank}_{r['name']}.png")
        shutil.copy2(src, dst)                      # untouched, full resolution
        im_w, im_h = _png_size(dst)
        print(f"  {dst.name:32} {im_w}x{im_h}  {dst.stat().st_size/1e6:.1f} MB  "
              f"grade {r['grade']:.1f}  {r['status']}")

        # The grade cannot see construction, so a pick can rank first and still
        # be a redrawn garment. Name the regions on the pick's own line - a
        # defect that only appears in the ranking table 40 lines up is a defect
        # nobody reads.
        mism = [c for c in r.get("construction", [])
                if c.get("verdict") == "MISMATCH"]
        if mism:
            shipped_bad.append((r["name"], [c["region"] for c in mism],
                                r.get("construction_from", "not checked")))
            print(f"  {'':4}SHIPPED WITH ALTERED CONSTRUCTION: "
                  f"{', '.join(c['region'] for c in mism)}")
            for c in mism:
                print(f"  {'':6}{c['region']}: {c.get('detail','')[:150]}")

        # The cutout puts the garment on its own layer, so the retouch team sets
        # placement, canvas and plate themselves. That is why nothing in this
        # pipeline grades framing - and it still does not, so a pick shipped
        # without one carries whatever framing the generator chose.
        if not args.cutout:
            continue
        import cutout
        co = outd / f"{dst.stem}_cutout.png"
        try:
            info = cutout.cut(dst, co, feather=0.0, trim=False, pad=24)
            print(f"  {'  -> ' + co.name:32} {info['size'][0]}x{info['size'][1]}"
                  f"  {info['mb']:.1f} MB  transparent background")
        except Exception as e:  # noqa: BLE001 - a failed cutout must not lose the pick
            print(f"  {'  -> ' + co.name:32} FAILED: {e}")

    short = len(picks) < args.deliver
    if short:
        rest = [r for r in rows if r not in picks]
        print(f"\nONLY {len(picks)} OF {args.deliver} SHIPPED. "
              f"{len(rest)} more are already generated and paid for.")
        print("Next best, in order - each with what it would cost you:")
        for r in rest[:max(args.deliver - len(picks) + 2, 4)]:
            # "grade N < mark" is already the note for a BELOW, so naming the
            # column "grade" too prints the number twice on the same line.
            why = [n for n in r["notes"] if not n.startswith("grade ")]
            print(f"  {r['name']:10} grade {r['grade']:6.1f}  {r['status']:<7} "
                  f"{'; '.join(why)[:70]}")
        print(f"Look at {arch.name}/grade_results.jpg, then copy the ones you "
              f"would defend into {outd.name}/ yourself and record in `## Notes` "
              f"what each carries. Do not generate more - if the batch failed "
              f"the same way repeatedly, the prompt is wrong and more draws buy "
              f"more of the same.")

    if shipped_bad:
        print(f"\n{len(shipped_bad)} of {len(picks)} shipped with construction "
              f"the vision check flagged as altered:")
        for name, regions, when in shipped_bad:
            print(f"  {name:10} {', '.join(regions):<40} {when}")
        print(f"Per-region detail is in {arch.name}/grade_results.json. Say in "
              f"`## Notes` which picks carry this - the grade does not, and "
              f"output/ on its own cannot tell anyone.")

    flags = "".join([f" [{args.ship_mode}]",
                     " (no cutouts)" if not args.cutout else "",
                     f" ({len(shipped_bad)} with altered construction)"
                     if shipped_bad else ""])
    C.log(args.run, f"shipped {len(picks)}{flags}: "
                    f"{','.join(r['name'] for r in picks)}")
    return 0 if not short else 2


def _png_size(p: Path) -> tuple[int, int]:
    out = subprocess.run(["magick", "identify", "-format", "%wx%h", str(p)],
                         check=True, capture_output=True, text=True).stdout
    w, _, h = out.partition("x")
    return int(w), int(h)


if __name__ == "__main__":
    sys.exit(main())
