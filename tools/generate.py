#!/usr/bin/env python3
"""The only billed step. One call per image, run concurrently.

    python tools/generate.py --run runs/<stamp> --num 5 --resolution 2K

One image per call with its own seed, so each is an independent sample rather
than one batch the server draws together - which is what a shortlist needs.
Concurrency keeps the wave inside the harness's 900s bash timeout; ten serial
generations measured ~595s on this project.

Numbering continues from what is already in the folder, so topping up needs no
arguments. `--max-total` is a hard ceiling on images per run, enforced here
because task text is advisory and has been overridden on real runs.

Candidates are numbered by SUBMISSION order, so cand_03 is always the third seed
however the wave happens to land.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

import common as C

# The operator's ceiling on images per run folder. Set it here, or per run with
# LAYDOWN_MAX_IMAGES=8 ./run.sh ...  --max-total can lower it but never raise it,
# because the agent chooses that flag and task text has been overridden on four
# separate runs.
HARD_CAP = int(os.environ.get("LAYDOWN_MAX_IMAGES", "5"))

# nano-banana-pro/edit takes a `system_prompt` alongside the scene prompt, and
# this project was not using it. That is the channel the duplicate-garment
# failure needed: the endpoint accepts a list of image_urls with no way to mark
# one of them as reference rather than content, so image 1 and image 2 arrive as
# two pictures that each contain a sweater and the model decides for itself what
# to do with the second one. Roughly two in ten came back holding both.
#
# Everything tried before this was scene-prompt persuasion, and the scene prompt
# is read as a description of the picture to make - which is why naming the
# failure there made it worse rather than better (8 in 10 on
# runs/20260827_110352). A system instruction is read as a standing rule about
# the job instead, so the role of each input can be stated once, up front, in
# the register where it belongs.
#
# Phrased positively throughout, for the reason recorded on the count clause
# below: naming the thing to avoid is how a diffusion model is told to draw it.
SYSTEM = (
    "You are a product photography retoucher. You produce catalogue flat lays: "
    "one garment, photographed from directly above, lying flat and square on a "
    "plain white background. "
    "The first input image is the product. Its construction, colour, pattern, "
    "texture and proportions are reproduced exactly as they are. "
    "Any further input image is a layout guide. It shows how the garment should "
    "be arranged and it is reference material rather than content: it "
    "contributes arrangement only, and its own subject stays out of the "
    "picture you produce. "
    "Every image you return is a single photograph of one garment."
)


# What a prompt has to say, and what it must not, with the run that paid for
# each rule. The skill has documented these refusals for some time; until now
# nothing enforced them, which is worse than not having them - a guardrail
# nobody can rely on still gets relied on.
MUST_MENTION = (
    ("flat", ("flat", "flatly", "flatness", "laid flat", "lay flat"),
     "the model adds volume and 3D shaping unless told not to"),
    ("wrinkles", ("wrinkle", "crease", "rumple", "fold line", "steamed",
                  "pressed"),
     "de-wrinkling is the job; it does not happen by default"),
    ("background", ("background", "backdrop", "plate", "white studio"),
     "an unmentioned backdrop comes back grey, textured or replaced"),
)

# Asking for transparency produces a painted checkerboard: the endpoint cannot
# output an alpha channel, and a real run came back with a literal grey-and-white
# checker pattern counted as 1587-3135 background specks. cutout.py adds real
# transparency locally, afterwards, for free.
FORBID_TRANSPARENT = ("transparent", "transparency", "alpha channel",
                      "remove the background", "background removed",
                      "removed background", "no background")

# The source is pre-cleaned before the prompt is written, so there is no tag
# left to keep or remove - and naming one invites the model to draw one. A run
# that described the raw input's hang tag and asked that it "stay in place" got
# four candidates that grew a tag which was not in the image sent.
FORBID_PAPERWORK = ("hang tag", "hangtag", "swing tag", "price ticket",
                    "ticket", "barcode", "hanger", "tag", "label")

# Naming a viewpoint is how a garment comes back flipped. The describe pass
# writes what it thinks it is looking at - "shown from the back", "two back
# panels forming the criss-cross straps" - the agent copies it into the prompt,
# and the model does as it is told: seven of ten candidates in one batch came
# back showing the reverse face, every seam correct and the garment inside out.
# The clause this script appends already fixes the face; nothing in the agent's
# own prompt needs to mention a side at all, so any mention is a risk with no
# upside.
FORBID_ORIENTATION = ("shown from the back", "shown from the front",
                      "from behind", "reverse side", "viewed from",
                      "back view", "front view", "rear view", "inside out",
                      "wrong side")

# Absolutes that ask for MORE than flat: they push the model past relaxing the
# handling folds and into repainting the fabric as a smooth surface, which is
# this project's documented redraw driver. A warning, not a refusal - the
# wording is a judgement call and the grade now measures the consequence
# directly, scoring texture distance from the source in both directions.
SOFTEN = ("wrinkle-free", "wrinkle free", "no creases", "no wrinkles",
          "completely smooth", "perfectly smooth", "freshly steamed",
          "steamed and pressed", "ironed", "no fold lines", "flawless")


def check_prompt(prompt: str, reference: Path,
                 min_words: int = 120) -> tuple[list[str], list[str]]:
    """(problems, warnings). Problems refuse the run; warnings only print."""
    low = prompt.lower()
    out, warn = [], []

    n = len(prompt.split())
    if n < min_words:
        out.append(f"{n} words, under the {min_words}-word floor. Short prompts "
                   f"leave the lay, the flatness and the construction to the "
                   f"model's own judgement, which is what is being replaced.")

    for name, words, why in MUST_MENTION:
        if not any(w in low for w in words):
            out.append(f"never mentions {name} - {why}")

    try:
        greyscale = Image.open(reference).mode == "L"
    except Exception:  # noqa: BLE001 - an unreadable reference fails later anyway
        greyscale = False
    if greyscale and not any(w in low for w in
                             ("greyscale", "grayscale", "grey", "gray", "tone",
                              "colourless", "colorless", "outline", "silhouette",
                              "line drawing")):
        out.append("never says image 2 is greyscale - read as a colour target, "
                   "it desaturates the garment")

    hits = [w for w in FORBID_TRANSPARENT if w in low]
    if hits:
        out.append(f"asks for a transparent or removed background "
                   f"({', '.join(hits)}) - the model cannot output alpha and "
                   f"paints a checkerboard into the pixels instead. Ask for a "
                   f"plain white plate; cutout.py adds transparency afterwards.")

    hits = [w for w in FORBID_PAPERWORK if re.search(rf"\b{re.escape(w)}\b", low)]
    if hits:
        out.append(f"mentions paperwork ({', '.join(hits)}) - the source is "
                   f"already cleaned, so there is none to keep or remove, and "
                   f"naming it makes the model draw one. If this garment has a "
                   f"genuine sewn-in label that must survive, --force says so "
                   f"deliberately.")

    hits = [w for w in FORBID_ORIENTATION if w in low]
    if hits:
        out.append(f"names a viewpoint ({', '.join(hits)}) - this is how a "
                   f"garment comes back flipped, and seven of ten candidates in "
                   f"one batch did. This script already appends a clause saying "
                   f"to show the same face as image 1; say nothing about sides "
                   f"yourself.")

    hits = [w for w in SOFTEN if w in low]
    if hits:
        warn.append(f"asks for absolute smoothness ({', '.join(hits)}). Relaxing "
                    f"handling folds is the job; ironing the knit out of "
                    f"existence is a redraw, and the grade scores texture "
                    f"DISTANCE from the source, so an over-smoothed candidate "
                    f"loses points in the same direction as a creased one. "
                    f"Prefer 'relax the folds so the fabric lies flat, keep the "
                    f"knit's real texture'.")
    return out, warn


def reference_risk(run: Path) -> dict:
    """What step 0 found the reference carrying beyond the lay, if anything.

    Written by select_reference.py into reference_selection.json before a penny
    is spent. Missing or unreadable is not an error - it means step 0 did not
    run, and the generic clause is sent on its own.
    """
    f = Path(run) / "reference_selection.json"
    if not f.exists():
        return {}
    try:
        sel = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    risk = dict(sel.get("construction_risk") or {})
    risk["silhouette"] = bool(sel.get("silhouette"))
    return risk


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=C.INPUTS / "reference_image.jpg")
    ap.add_argument("-n", "--num", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=5,
                    help="calls in flight. Lower if fal rate-limits.")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="image i uses base_seed + i, recorded in seeds.json")
    ap.add_argument("--seed", type=int,
                    help="exact seed for this call. With --num 1 this is how "
                         "you re-roll a specific draw, or deliberately keep one "
                         "while the prompt changes around it.")
    ap.add_argument("--resolution", default="4K", choices=["1K", "2K", "4K"])
    # Shared with prepare.py, which letterboxes the source to this same shape.
    # If they disagree the padding stops working and the ghost comes back.
    ap.add_argument("--aspect-ratio", default=C.ASPECT)
    ap.add_argument("--max-total", type=int, default=HARD_CAP,
                    help=f"images allowed per run folder, counting everything "
                         f"already on disk. Currently {HARD_CAP}. Cannot be "
                         f"raised above LAYDOWN_MAX_IMAGES ({HARD_CAP}) - that "
                         f"is the operator's ceiling, not the agent's.")
    ap.add_argument("--force", action="store_true",
                    help="generate anyway when the prompt looks incomplete")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the full prompt - yours plus the "
                         "construction inventory and the image-2 clause this "
                         "script appends - print it, and stop before anything "
                         "is uploaded or billed. The only way to read what the "
                         "model will actually receive without paying for it.")
    ap.add_argument("--ignore-clean-gate", action="store_true",
                    help="generate from a source the pre-clean gate REJECTED. "
                         "This is a decision to pay for images of a garment "
                         "that is already known to be damaged; it is recorded "
                         "in steps.log as such.")
    a = ap.parse_args()

    run = a.run
    arch = run / "archive"
    src = arch / "offset_upload.jpg"
    prompt_file = arch / "prompt.txt"
    if not src.exists():
        return print(f"Not found: {src}. Run prepare.py first.") or 1
    if not prompt_file.exists():
        return print(f"No {prompt_file}.\n"
                     f"prepare.py does not write one - look at both images, "
                     f"then write it yourself. See "
                     f"{arch / 'prompt_brief.md'}.") or 1
    if not a.reference.exists():
        return print(f"Not found: {a.reference}") or 1

    # The last free moment. Everything below this line is billed, and there is
    # no point buying ten independent draws of a garment the pre-clean step
    # already reported as damaged - they all inherit the damage, and every check
    # downstream then agrees the garment always looked like that.
    clean = C.clean_verdict(run)
    if clean["checked"] and not clean["ok"] and not a.ignore_clean_gate:
        print("REFUSING TO GENERATE: the pre-clean gate rejected this source.")
        for f in clean["fails"]:
            print(f"  - {f}")
        print(f"  audit: {arch / 'clean_audit.json'}")
        print("  The cleaned upload is on disk so it can be looked at, but the "
              "garment in it is not the garment in the photograph. A run that "
              "generated from one of these spent its whole image budget "
              "(150c) on a source that was never usable, and the pins the "
              "clean failed to remove then showed up as a construction "
              "MISMATCH on all ten candidates.")
        print("  Fix the clean - or --ignore-clean-gate to pay for it anyway, "
              "on the record.")
        C.log(run, f"generation REFUSED: pre-clean gate failed ({clean['fails'][0][:40]})")
        return 1
    if clean["checked"] and not clean["ok"]:
        print(f"--ignore-clean-gate: generating from a source the gate rejected "
              f"({clean['why'][:80]})")

    prompt = prompt_file.read_text()

    # Checked before the money, on the agent's own text only - the blocks
    # appended below are this script's and are not the agent's to get wrong.
    complaints, warnings = check_prompt(prompt, a.reference)
    for w in warnings:
        print(f"prompt WARNING: {w}")
    if complaints and not a.force:
        print(f"REFUSING TO GENERATE: {prompt_file} is not ready.")
        for cpt in complaints:
            print(f"  - {cpt}")
        print(f"  Fix {prompt_file.name} and run again, or --force to send it "
              f"as it stands. See {arch / 'prompt_brief.md'}.")
        C.log(run, f"generation REFUSED: prompt ({len(complaints)} problem(s), "
                   f"{complaints[0][:40]})")
        return 1
    if complaints:
        print(f"--force: sending a prompt with {len(complaints)} unresolved "
              f"problem(s)")
        for cpt in complaints:
            print(f"  - {cpt}")

    # The construction inventory rides on EVERY prompt, automatically. It is
    # measured from the cleaned source by a hosted vision model, and it is the
    # anchor against invented seams - especially its NOT-PRESENT section. It is
    # appended rather than left to the agent because instructions in this skill
    # have been skipped on four separate runs.
    # What gets injected depends on where the construction CAME from.
    #
    # From a spec sheet, both halves ship: it is authored ground truth, and its
    # negative statements are the strongest thing available - this project's
    # sheet says "No inseam with oval gusset", which is precisely the seam two
    # different VLMs invented when guessing from the photograph.
    #
    # From the photograph, only the NOT-PRESENT half ships. The positive list is
    # open-ended recall and fabricated on all five attempts, at both model
    # tiers; sending it under "reproduce exactly this" is what put a topstitched
    # seam down each leg of every candidate in a real run.
    desc_file = arch / "garment_description.md"
    if desc_file.exists():
        txt = desc_file.read_text()
        up = txt.upper()
        grounded = "spec sheet, transcribed" in txt[:200]

        # Every NOT-PRESENT claim becomes "the garment specifically does NOT
        # have this" in the prompt, so a false one tells an image model to
        # delete real construction. They are audited here, at the last free
        # moment, rather than trusted: on a real run the list named "Pearl
        # embellishments" while the same document described four of them, and
        # named "Racerback straps" and "Pullover style" for a garment step 0
        # had measured as exactly those. All four were sent.
        import describe
        audit = describe.audit_absent(txt, describe.query_attrs(run))
        absent = ", ".join(audit["keep"])
        if audit["dropped"]:
            print(f"construction inventory: {len(audit['dropped'])} "
                  f"NOT-PRESENT claim(s) withheld as contradicted")
            for item, why in audit["dropped"]:
                print(f"  {item:<32} {why}")

        block = ["\n\n---"]
        if grounded and "**CONSTRUCTION**" in txt:
            spec = txt.split("**CONSTRUCTION**", 1)[1]
            spec = spec.split("**NOT PRESENT", 1)[0].strip()
            # describe.py already strips these at write time; doing it again
            # here costs nothing and covers a description written before that
            # existed, or edited by hand since.
            spec, dropped_orient = describe.strip_orientation(spec)
            for s in dropped_orient:
                print(f"  orientation claim withheld from the prompt: {s[:80]}")
            block += [
                "CONSTRUCTION SPEC, taken from this product's own spec sheet. "
                "This is what the garment is genuinely built with:", "", spec, "",
                "This lists what EXISTS on the garment, not what is visible from "
                "this side. Reproduce only the construction you can actually see "
                "in image 1 - do not add a seam or panel from this list that is "
                "not visibly there."]
        else:
            block += ["The garment has NO construction beyond what is visible in "
                      "image 1. Reproduce only the seams, panels and details you "
                      "can actually see - add no seam, panel line, topstitching "
                      "or feature that is not visibly present."]
        if absent:
            block += ["", f"It specifically does NOT have: {absent}"]

        # What the pre-clean used to erase before the model ever saw it. That
        # step no longer runs - fal's object-removal endpoint is restricted -
        # so image 1 still contains the tag, the pins and the hanger, and the
        # only thing standing between them and the output is this sentence.
        # Stated as objects to omit rather than as "clean the image", because
        # a named list is the form these models act on most reliably.
        gone = describe.removals(txt)
        if gone:
            block += ["",
                      "Image 1 has NOT been retouched. These are in the photo "
                      "but are NOT part of the garment, and none of them may "
                      "appear in the output: " + "; ".join(gone) + ".",
                      "Remove them completely and reconstruct the garment "
                      "underneath - no tag, no pin, no clip, no hanger, no "
                      "outline, shadow or gap where one used to be."]
            print(f"prompt asks for {len(gone)} non-garment item(s) to be "
                  f"removed: {', '.join(g[:40] for g in gone)}")
        prompt = prompt.rstrip() + "\n".join(block) + "\n"
        print(f"prompt carries construction "
              f"{'from the SPEC SHEET (authoritative)' if grounded else 'inferred from the photo (NOT-PRESENT only)'}"
              + (f"; absent: {absent[:60]}" if absent else ""))

    # The lay reference is image 2, and the model cannot tell which parts of it
    # are "how this should be arranged" and which are "how that other product is
    # built". So every prompt says it, and when step 0 found the reference
    # differing in construction terms, this names the exact words it found.
    #
    # It is appended here rather than left to the prompt the agent writes,
    # because the brief has been skipped on four separate runs and because step
    # 0 measured this - the agent is not in a position to know it.
    # The pose half of this clause had to be made explicit and made to WIN.
    # Saying image 2 is the lay reference is not enough on its own: the prompts
    # the agent writes all carry "keep the garment exactly as in image 1",
    # "untouched", "adding nothing", repeated and emphatic, and the model reads
    # that as covering pose as well as construction. On runs/20260827_101946 the
    # source has its sleeves angled inward with a bend at the elbow, image 2 has
    # them straight down at the sides, and most candidates came back with the
    # SOURCE pose - the reference was out-argued by the louder instruction.
    #
    # So the split is now stated as a rule with a winner named for each half:
    # image 1 owns what the garment IS, image 2 owns how it LIES, and where they
    # disagree about lying, image 2 wins. Without the last part the model has no
    # way to resolve the conflict and defaults to changing the least.
    risk = reference_risk(run)
    clause = ["\n\n---",
              # PHRASED POSITIVELY, ON PURPOSE. This clause used to spell the
              # failure out - "no second copy, no partial copy, no collar or
              # cuff of another garment, however faint" - and the duplicate rate
              # went UP, from 2 in 10 on runs/20260827_101946 to 8 in 10 on
              # runs/20260827_110352, the worst measured. Both clauses were
              # verified present in the sent prompt, so it was not ignored.
              #
              # Naming the thing eight times is how a diffusion model is told to
              # draw it. Negation is not reliably represented: the tokens that
              # land are "second copy", "another garment", "collar", and the
              # "no" in front of them carries far less weight than it does in
              # conversation. So the count is now stated once, as a fact about
              # what the frame contains, and the word for the failure never
              # appears.
              "The frame contains one garment, centred, surrounded by empty "
              "white background on all sides.",
              "",
              "IMAGE 2 IS A LAY REFERENCE ONLY. It shows how the garment should "
              "be ARRANGED - how it lies, how straps or legs are positioned, how "
              "square and symmetric the lay is. Take NOTHING else from it. Every "
              "seam, stitch line, neckline shape, panel, binding, trim and piece "
              "of hardware comes from image 1 and only from image 1. If image 2 "
              "shows construction image 1 does not have, it belongs to a "
              "different product: do not reproduce it.",
              "",
              # Also positive, for the same reason as the count above. This
              # named the poses to avoid - "creased, angled, arms folded or
              # turned in" - which is a list of things to draw as far as a
              # diffusion model is concerned. It describes the target pose only,
              # and settles the conflict by naming which image wins rather than
              # by describing what image 1 got wrong.
              # DESCRIBED IN WORDS, not by pointing at image 2. The first
              # version of this clause said "MATCH THE POSE IN IMAGE 2 ... image
              # 2 is the authority ... follow image 2 every time", and the
              # duplicate rate tracked it exactly: every run carrying it came in
              # at 60-80% (runs/20260827_104532, _110352, _111218), against
              # 20-44% on the two runs before it existed (_101946, _095355).
              # The extra garments are reported by the grader as "faded,
              # semi-transparent" - reference-like.
              #
              # That is the clause doing its job too well. Told to match image 2
              # and that image 2 wins, the model reproduces image 2 - and image
              # 2 contains a garment, so a second garment is drawn. The old
              # layout-only clause below already establishes what image 2 is
              # for; this one only needs to say what the finished lay looks
              # like, which it can do without naming the picture at all.
              #
              # TRIED AND REVERTED: a sentence describing the body outline -
              # "the body lies open and flat to its full width, each side seam
              # pulled straight from underarm to hem so the outline down each
              # side is one clean continuous line". It was added because this
              # clause names the sleeves, the shoulders, the hem and the tilt
              # but never the body, and on runs/20260827_122117 every candidate
              # kept the SOURCE's outline, bowed out at the stomach, while the
              # reference lies with its sides straight.
              #
              # It did not work and it was not free. Two runs carried it -
              # _123706 at 7/10 duplicates and _124623 at 6/10, against 2-6 on
              # the six runs before it - and the second of those had both
              # suspected confounders removed (the agent's prompt no longer
              # called image 2 "a different garment", and the sentence no longer
              # ended on the word "mirror"), so neither of those explains the
              # rate. Meanwhile it bought nothing measurable: mean silhouette
              # IoU against the source over the clean candidates of the three
              # runs on this garment went 0.849 without it, 0.813, then 0.849
              # with it. Same outline, more ghosts.
              #
              # The lesson is the one already written above, and it applies to
              # the garment as well as to image 2: every additional sentence
              # about the garment's own body is another mention of the subject,
              # and this model answers a repeated subject with a second copy.
              # Straightening the stomach has to come from somewhere other than
              # more prompt - the agent's own "must not stretch or slim it"
              # line, and the fact that grade_flats scores silhouette against
              # the source, both pin the shape to image 1.
              "LAY THE GARMENT SQUARE AND FLAT. Sleeves or legs straight down "
              "at the sides, evenly spaced and symmetric, each cuff clear of "
              "the body. Shoulders level, hem level and parallel to the bottom "
              "of the frame, the garment upright and centred with no rotation "
              "or tilt. This is how the finished flat lies, whatever "
              "arrangement the garment happens to be in when photographed.",
              "",
              # Keeps the scoping that fixed the folded-arm problem - "keep it
              # unchanged" was being read as covering pose - without pointing at
              # image 2 as something to reproduce. The re-lay is now framed as
              # the task itself rather than as copying a picture.
              "Any instruction above to keep the garment unchanged, untouched or "
              "exactly as in image 1 refers to its CONSTRUCTION, COLOUR, PATTERN "
              "and TEXTURE only. It never refers to the pose. Re-laying the "
              "garment flat and square IS the task, so moving a sleeve, "
              "straightening a hem or squaring the body is required, not a "
              "change to be avoided.",
              "",
              # The flip is not a construction error and no construction clause
              # prevents it: every seam can be correct and the garment still
              # arrive inside-out or back-to-front, because the lay reference
              # shows a different face and the model reconciles the two.
              "Show the same face of the garment as image 1, exterior toward the "
              "camera - do not flip the garment or show its interior or reverse "
              "side. Image 2 defines only how flat and square the garment lies, "
              "never which side faces up."]
    if risk.get("flagged"):
        clause += ["",
                   f"Specifically, image 2 was measured as differing from this "
                   f"product in: {', '.join(risk['terms'])}. "
                   + (f"What was observed: {risk['line']} " if risk.get("line") else "")
                   + "None of that is on the product in image 1. Do not add it."]
    # The blunt closer that used to live here - "ONE GARMENT ONLY. RETURN A
    # SINGLE GARMENT IN THE IMAGE." - is gone with the rest of the negation. It
    # was the last thing the model read and it repeated the subject one more
    # time. The count is stated once, positively, above.
    prompt = prompt.rstrip() + "\n".join(clause) + "\n"
    if risk.get("flagged"):
        print(f"prompt hardened against reference bleed: "
              f"{', '.join(risk['terms'])}"
              + ("  (reference is an outline map, so there is nothing to copy)"
                 if risk.get("silhouette") else ""))
    else:
        print("prompt carries the layout-only clause for image 2")
    print("prompt makes image 2 authoritative for pose - sleeve angle, cuff "
          "spacing, hem line and centring follow the reference, not the source")

    if a.dry_run:
        print(f"\n--- the full prompt as sent ({len(prompt.split())} words) "
              f"{'-' * 30}\n")
        print(prompt)
        print("-" * 74)
        print("DRY RUN - nothing uploaded, nothing billed, no snapshot written.")
        return 0

    C.fix_ca_bundle()
    C.load_fal_key()
    import fal_client
    import requests

    print("Uploading both inputs once, reused by every call...")
    src_url = fal_client.upload_file(str(src))
    ref_url = fal_client.upload_file(str(a.reference))

    # Numbering continues from whatever is already here, so a top-up needs no
    # bookkeeping. Every image is a candidate: at one resolution there is no
    # such thing as a throwaway probe, and treating some as throwaway once
    # discarded the three best images of a run.
    cap = min(a.max_total, HARD_CAP)
    if a.max_total > HARD_CAP:
        print(f"--max-total {a.max_total} exceeds the operator ceiling of "
              f"{HARD_CAP}; using {HARD_CAP}. Raise LAYDOWN_MAX_IMAGES to "
              f"change it.")
    existing = sorted(int(p.stem.split("_")[1]) for p in arch.glob("cand_*.png"))
    have = len(existing)
    room = cap - have
    if room <= 0:
        print(f"Run already holds {have} image(s), at the --max-total ceiling "
              f"of {cap}. Measure and pick from those, or start a new "
              f"run folder.")
        return 1
    n = min(a.num, room)
    if n < a.num:
        print(f"Asked for {a.num}, but {have} of {cap} are already "
              f"here - generating {n}.")

    start = (max(existing) + 1) if existing else 1
    nums = list(range(start, start + n))
    if a.seed is not None:
        seeds = {i: a.seed + (i - start) for i in nums}
    else:
        seeds = {i: a.base_seed + i for i in nums}
    name = lambda i: arch / f"cand_{i:02d}.png"

    # Snapshot the prompt under its own hash. prompt.txt is a working file the
    # agent rewrites between calls, so without this a run cannot say which
    # wording produced which image - and answering that took reading file
    # mtimes on a real run. Identical prompts reuse one snapshot.
    ph = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    snap = arch / f"prompt_{ph}.txt"
    if not snap.exists():
        # The system instruction is sent on every call and is not part of the
        # scene prompt, so it is recorded here too. Without it a snapshot only
        # accounts for half of what the model was told, and reading back a run
        # to work out why it behaved as it did is most of what this file is for.
        # Appended after the prompt, and NOT part of `ph`, so changing it does
        # not silently rewrite the identity of an existing prompt.
        snap.write_text(prompt.rstrip() + "\n\n<!-- system_prompt sent with "
                        "every call in this run:\n" + SYSTEM + "\n-->\n")
        print(f"prompt {ph} ({len(prompt.split())} words) -> {snap.name}")
        print(f"       + system instruction ({len(SYSTEM.split())} words): "
              f"image 2 declared reference-not-content, one garment per image")
    else:
        print(f"prompt {ph} (unchanged)")

    def one(i: int):
        args = {"prompt": prompt, "image_urls": [src_url, ref_url],
                "num_images": 1, "output_format": "png",
                "system_prompt": SYSTEM,
                "resolution": a.resolution, "aspect_ratio": a.aspect_ratio,
                "seed": seeds[i]}
        for attempt in (1, 2):
            try:
                r = fal_client.subscribe(C.ENDPOINT, arguments=args, with_logs=False)
                items = r.get("images", [])
                if not items:
                    raise RuntimeError("no images returned")
                img = Image.open(requests.get(items[0]["url"], stream=True,
                                              timeout=300).raw).convert("RGB")
                return i, img, None
            except Exception as e:
                if attempt == 2:
                    return i, None, str(e)
                time.sleep(3)
        return i, None, "unreachable"

    print(f"{a.num} calls to {C.ENDPOINT} at {a.resolution} {a.aspect_ratio}, "
          f"{a.concurrency} at a time")
    got, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as pool:
        futs = [pool.submit(one, i) for i in nums]
        for f in as_completed(futs):
            i, img, err = f.result()
            if err:
                print(f"  [{i:2}] FAILED twice: {err}", flush=True)
                continue
            p = name(i)
            img.save(p)
            got.append(i)
            print(f"  [{i:2}] {p.name}  {img.width}x{img.height}  "
                  f"+{time.time()-t0:.0f}s", flush=True)

    # Merge rather than overwrite: a split batch calls this twice, and the
    # second call must not erase the first call's records.
    #
    # Resolution is recorded per image because it decides eligibility later. A
    # probe generated at the SAME resolution as the final batch is a candidate
    # in every respect - same model, same prompt, same pixels - and discarding
    # it because of its filename threw away the three best images of a real run.
    sf = arch / "seeds.json"
    try:
        rec = json.loads(sf.read_text()) if sf.exists() else {}
    except json.JSONDecodeError:
        rec = {}
    rec.update({name(i).stem: {"seed": seeds[i], "resolution": a.resolution,
                               "prompt": ph} for i in sorted(got)})
    sf.write_text(json.dumps(rec, indent=2))

    cents = len(got) * (C.PRICE_4K if a.resolution == "4K" else 0.15) * 100
    total = have + len(got)
    gate_note = (" FROM A SOURCE THE CLEAN GATE REJECTED"
                 if clean["checked"] and not clean["ok"] else "")
    print(f"\n{len(got)}/{n} in {time.time()-t0:.0f}s   "
          f"run now holds {total}/{cap}")
    if len(got) < n:
        print("  WARNING: short. Do NOT re-run to top up - every image already "
              "on disk is already billed. Measure what landed.")
    C.log(run, f"generated {len(got)} at {a.resolution}, prompt {ph} "
               f"({total} total){gate_note}", cents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
