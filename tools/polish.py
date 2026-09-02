#!/usr/bin/env python3
"""The cleanup pass: take the wrinkles out of a finished candidate, change nothing else.

    python tools/polish.py --run runs/<stamp> --candidate cand_04
    python tools/polish.py --run runs/<stamp> --candidate cand_04 \
        --instruction "keep the garment as is, just relax the creases"

A second, different model - `openai/gpt-image-2/edit` - run on ONE image with a
narrow instruction. It is not another generation: nano-banana re-lays the garment
and this only smooths what is left, which is why it takes no reference and gets
its own small budget.

WHY A DIFFERENT MODEL. This is a targeted edit rather than a re-lay, and
gpt-image-2/edit is built for exactly that shape of job - "apply this change,
leave everything you were not asked about alone". Asking nano-banana for one more
pass instead re-rolls the whole picture: the documented behaviour on this project
is that a corrective second pass comes back worse, having added texture that was
never there.

THE RISK, NAMED. De-wrinkling is the one instruction on this project with a
measured way of going too far. "Wrinkle-free", "completely smooth" and friends do
not stop at relaxing handling folds; they repaint the knit as a smooth surface,
which is a redraw wearing a tidy face. So the default instruction says to keep
the fabric's texture in the same breath as removing the creases, and every result
is measured against the ORIGINAL source before it is handed back. A wrinkle ratio
that has collapsed is the signature of an ironed garment, and it is printed as a
warning rather than left for someone to notice.

BILLING IS NOT LIKE THE REST OF THE PIPELINE. nano-banana is a flat rate per
image. gpt-image-2 bills by token, so the cost of a call depends on the size and
quality of what comes back and cannot be known in advance. The figure logged here
is an ESTIMATE at published rates, flagged as one - fal exposes no billing API,
so no number in this project is ever a receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import generate as G  # noqa: E402
import metrics as M  # noqa: E402

# Confirmed against fal's own model page. Overridable because a model slug is the
# thing most likely to move, and a wrong one should be a one-line fix rather than
# a code change.
ENDPOINT = os.environ.get("POLISH_ENDPOINT", "openai/gpt-image-2/edit")

# Polishes per run. Small and separate from the image budget on purpose: this is
# a finishing move on a candidate that is already good, and a run that needs five
# of them has a problem the polish is not going to fix.
MAX_POLISH = int(os.environ.get("LAYDOWN_MAX_POLISH", "3"))

# Token-billed, so this is an estimate and is labelled as one everywhere it is
# printed. Roughly one high-quality edit at the published image-token rate.
EST_CENTS = 15.0

# The operator's words, kept close to how they were given, with one clause added.
# "Just the wrinkles" is already the whole instruction; the texture sentence is
# there because this specific request is the one that has historically been
# over-obeyed, and losing the knit is not a smaller failure than keeping a crease.
DEFAULT_INSTRUCTION = (
    "Keep the garment exactly as it is and remove only the wrinkles. Relax the "
    "creases and handling folds in the fabric so it lies smooth and flat. "
    "Change nothing else: not the shape, the colour, the proportions, the "
    "seams, the stitching, the trim, the buttons or zipper, the labels, the "
    "logos or the pattern. "
    # Named explicitly after the first live polish pulled the sleeves inward and
    # shrank the garment in frame - IoU 0.833 against its own parent. The pose is
    # the most expensive thing to get right on this project and the easiest for a
    # cleanup pass to quietly undo, so "position in frame" was not enough: the
    # arrangement has to be spelled out the same way the pose section spells it.
    "Do not move anything: the sleeves stay at exactly the angle they are now, "
    "the cuffs stay where they are, the hem stays where it is, and the garment "
    "keeps the same size and position within the frame. "
    "Keep the fabric's real surface - the knit, weave or pile stays visible "
    "exactly as it is now, and is not smoothed into a flat painted surface. "
    "The background stays the same plain white it already is."
)


def polished_name(candidate: str) -> str:
    """cand_04 -> cand_04p. Sits alongside cand_04s for the segmented form."""
    return f"{candidate}p"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--candidate", default=None,
                    help="a candidate name. Defaults to whatever pick_best named.")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--quality", default="high",
                    choices=("auto", "low", "medium", "high"),
                    help="gpt-image-2 bills by token, so this moves the cost")
    ap.add_argument("--max-total", type=int, default=MAX_POLISH)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be sent and bill nothing")
    a = ap.parse_args()

    run = a.run
    arch = run / "archive"
    arch.mkdir(parents=True, exist_ok=True)

    candidate = a.candidate
    if not candidate:
        bf = arch / "best.json"
        if bf.exists():
            try:
                candidate = json.loads(bf.read_text()).get("candidate")
            except (json.JSONDecodeError, OSError):
                candidate = None
    if not candidate:
        print("No candidate given and nothing is marked as best. Pass "
              "--candidate cand_NN, or call pick_best first.")
        return 1

    src = G.resolve_source(run, candidate)
    if not src.exists():
        have = sorted(p.stem for p in arch.glob("cand_*.png"))
        print(f"Not found: {src.name}\n  candidates on disk: "
              f"{', '.join(have) or '(none)'}")
        return 1

    out_name = polished_name(candidate)
    out = arch / f"{out_name}.png"

    if a.dry_run:
        print(f"endpoint    {ENDPOINT}")
        print(f"image       {src.name}")
        print(f"quality     {a.quality}")
        print(f"would write {out.name}")
        print("\ninstruction:")
        print(f"  {a.instruction}")
        print("\n--dry-run: nothing uploaded, nothing billed.")
        return 0

    done = sorted(arch.glob("cand_*p.png"))
    if len(done) >= a.max_total:
        print(f"REFUSING TO POLISH: {len(done)}/{a.max_total} polishes used. "
              f"This is a finishing move, not a way to keep trying.")
        return 1

    C.fix_ca_bundle()
    C.load_fal_key()
    import fal_client
    import requests

    try:
        url = G.upload_cached(run, src, fal_client)
    except RuntimeError as e:
        print(f"REFUSING TO POLISH: {e}")
        return 1

    args = {"prompt": a.instruction, "image_urls": [url],
            "num_images": 1, "output_format": "png",
            "quality": a.quality, "image_size": "auto"}

    print(f"polishing {candidate} with {ENDPOINT} (quality {a.quality})")
    try:
        r = fal_client.subscribe(ENDPOINT, arguments=args, with_logs=False)
        items = r.get("images", [])
        if not items:
            raise RuntimeError("no images returned")
        img = Image.open(requests.get(items[0]["url"], stream=True,
                                      timeout=300).raw).convert("RGB")
    except Exception as e:  # noqa: BLE001 - the model reads this, not a traceback
        detail = str(e)
        hint = ""
        if "404" in detail or "not found" in detail.lower():
            hint = (f"\n  '{ENDPOINT}' was not accepted. Set POLISH_ENDPOINT to "
                    f"the current slug from fal's model page.")
        print(f"POLISH FAILED: {type(e).__name__}: {detail[:200]}{hint}")
        return 1

    img.save(out)

    # A polish is a hop away from the real garment like any other generative
    # pass. gpt-image-2 is asked to change one thing, but it CAN invent, and a
    # depth that flattered it would be the one number nobody could trust.
    parent_depth = G.depth_of(run, candidate)
    G.record(run, out_name, {
        "parent": candidate, "parent_kind": "polish",
        "depth": parent_depth + 1, "prompt_hash": None, "seed": None,
        "endpoint": ENDPOINT, "quality": a.quality,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    C.log(run, f"polished {candidate} -> {out_name} via {ENDPOINT} "
               f"(~{EST_CENTS:.0f}c est, token-billed)", EST_CENTS)

    print(f"  saved {out.name}")

    # The whole point of measuring here: this is the one instruction that goes
    # too far in a specific, detectable direction.
    # MEASURED AGAINST THE PARENT, not against the original source, and this is
    # the one place in the project where that is right. Everywhere else the
    # question is "is this still the same garment as the photograph", so the
    # original is the only honest baseline. Here the instruction was narrower:
    # "identical to cand_04 except the wrinkles". The parent IS the contract, and
    # comparing to the source instead conflates what the generation changed with
    # what the polish changed.
    #
    # The first run of this measured against the source, reported dE 1.3 -> 9.7,
    # and concluded "creases relaxed, texture still there" because it only ever
    # looked at the wrinkle ratio. Against the parent the same pair reads IoU
    # 0.833 / dE 1.7: colour held, outline moved. Only the second reading says
    # anything about whether the polish did as it was told.
    try:
        drift = M.compare(src, out)
        print()
        print("  vs its parent " + M.line(drift, out_name))
        iou, de, ratio = (drift["silhouette_iou"], drift["colour_de"],
                          drift["wrinkle_ratio"])

        # Scale is measured SEPARATELY because IoU cannot see it. silhouette_iou
        # normalises both masks to their own bounding boxes, so re-framing is
        # invisible to it: two polishes of this same parent scored 0.833 and
        # 0.835 while one had visibly shrunk the garment in frame and the other
        # had not. Without this line a cleanup that rescaled the product would
        # pass silently.
        parent_area = float(M.compare(src, src)["area_pct"])
        child_area = float(drift["area_pct"])
        scale = (child_area / parent_area) if parent_area > 0 else None

        # WARNINGS are only for the unambiguous failures. There are two polishes
        # of measured experience behind this file, so any threshold finer than
        # "obviously wrong" would be a guess dressed as a calibration - and a
        # guessed threshold that fires on a good result teaches everyone to
        # ignore the warning. The shape and scale numbers are printed every time
        # and left to the eye, which is the standing rule in this project anyway.
        # Surface energy at two scales, against the parent. This is what tells a
        # relaxed crease from a repainted knit: the job is meant to move `broad`
        # and leave `fine` roughly where it is. A single window cannot separate
        # them, which is why the ratio above passed a polish that had visibly
        # softened the fleece.
        fine = drift.get("wrinkle_fine_ratio")
        broad = drift.get("wrinkle_broad_ratio")

        # And the same fine reading against the ORIGINAL source. Everything else
        # in this block is deliberately parent-relative - the polish's contract
        # is with its parent - but texture is the one quantity that STACKS. On
        # runs/20260901_140946 generation spent 31% of the source's knit and the
        # polish took 12% of what was left; each step was unremarkable against
        # its own parent and the shipped file kept 61% of the fabric. Only a
        # reading against the photograph sees that, so it is measured here and
        # reported separately rather than folded into the parent numbers.
        fine_vs_source = None
        origin = arch / "source_clean.jpg"
        if origin.exists():
            try:
                fine_vs_source = M.compare(origin, out).get("wrinkle_fine_ratio")
            except Exception as e:  # noqa: BLE001 - a check is not the delivery
                print(f"  (could not measure texture against the source: {e})")

        # Two lists. `verdicts` are contract breaches and stop the polish from
        # shipping; `notes` are printed and recorded and change nothing. The
        # split moved on 2026-09-02, when the deliverable became the
        # reference's lay first: a garment ironed completely flat is now the
        # job, so it cannot be a reason to ship the creased parent instead.
        # What still breaks the contract is the pass changing the SHAPE -
        # re-framing the garment - painting the fabric itself out from under
        # the folds, or shifting the colour, which nothing downstream puts
        # back.
        verdicts = []
        notes = []
        if ratio is not None and ratio < 0.5:
            notes.append(
                f"ironed flat: wrinkle ratio {ratio:.2f} against its parent. "
                f"That is the job now - check the construction survived")

        # SHAPE OF THE DATA, not a magnitude. Broad falling while fine holds is
        # the pass working; fine falling meaningfully faster than broad is the
        # knit being painted out from under the fold shadows. 0.85 is one sixth
        # of a scale's worth of divergence - loose, because the two bands track
        # each other within a few percent on every candidate measured so far
        # (cand_04p: fine 0.885, broad 0.863, quotient 1.03) and anything that
        # parts them by more than a token amount is not sampling noise.
        if fine and broad and broad > 0 and (fine / broad) < 0.85:
            verdicts.append(
                f"KNIT PAINTED OUT: fine texture fell to x{fine:.2f} while the "
                f"creases only fell to x{broad:.2f}. The pass smoothed the "
                f"fabric itself, not the folds in it")

        # ANCHORED ON ONE RUN, and worth saying so. The only measured points are
        # runs/20260901_140946, where 0.61 against the source was called too
        # heavy by eye and its unpolished parent at 0.69 was acceptable. 0.65
        # sits between them. Treat it as the line that flags "go and look",
        # not as a calibration - it will move once there are more runs behind it.
        if fine_vs_source is not None and fine_vs_source < 0.65:
            notes.append(
                f"texture spent: the polished file keeps x{fine_vs_source:.2f} "
                f"of the ORIGINAL photograph's fabric texture")
        if de > 3.0:
            verdicts.append(
                f"COLOUR SHIFTED: dE {de:.1f} against its parent, which the "
                f"instruction ruled out")
        if scale is not None and not 0.80 <= scale <= 1.25:
            verdicts.append(
                f"RE-FRAMED: the garment covers {child_area:.1f}% of the frame "
                f"against the parent's {parent_area:.1f}% ({scale:.2f}x)")

        # Machine-readable, so the harness's ship step can decide which of the
        # two files to deliver without re-deriving any of this or parsing stdout.
        (arch / "last_polish.json").write_text(json.dumps({
            "child": out_name, "parent": candidate,
            "silhouette_iou": iou, "colour_de": de, "wrinkle_ratio": ratio,
            "wrinkle_fine_ratio": fine, "wrinkle_broad_ratio": broad,
            "wrinkle_fine_vs_source": fine_vs_source,
            "scale": scale, "verdicts": verdicts, "notes": notes,
            "broke_contract": bool(verdicts),
        }, indent=2) + "\n")

        if verdicts:
            print("\n  WARNING: the polish broke its own instruction.")
            for v in verdicts:
                print(f"    - {v}")
        for v in notes:
            print(f"  note: {v}")
        line = f"\n  shape  IoU {iou:.3f} vs parent"
        if scale:
            line += f"   scale {scale:.2f}x"
        if ratio:
            line += f"   texture x{ratio:.2f}"
        print(line)
        # Printed every time, warning or not. The standing rule on this project
        # is that the numbers go on screen and the eye decides; a threshold that
        # only speaks when it fires hides the run that sat just under it.
        if fine and broad:
            print(f"  fabric knit x{fine:.2f}, creases x{broad:.2f} vs parent"
                  + (f"   |   knit x{fine_vs_source:.2f} vs the original photo"
                     if fine_vs_source is not None else ""))
        # Three polishes of evidence so far, and they separate cleanly: two at
        # IoU ~0.834 where the sleeves and the framing had visibly moved, and one
        # at 0.994 / scale 1.00 that changed nothing but the creases. So a clean
        # polish CAN hold the outline almost exactly, and a reading in the 0.8s
        # is drift rather than measurement noise. Still reported rather than
        # enforced - three is not a calibration - but the gap is wide enough to
        # read at a glance.
        if iou >= 0.97 and scale and 0.95 <= scale <= 1.05:
            print("  Held its shape and framing almost exactly - this is what a "
                  "clean polish looks like.")
        else:
            print("  A clean polish holds IoU around 0.99 at scale 1.00; "
                  "readings in the 0.8s have meant the sleeves or the framing "
                  "moved. LOOK AT THE TWO SIDE BY SIDE - a polish that also "
                  "changed the pose reads better on its own and worse against "
                  "the reference.")
        print(f"  {candidate} is untouched on disk and is still shippable.")
    except Exception as e:  # noqa: BLE001 - a measurement is not the delivery
        print(f"  (could not measure against the parent: {type(e).__name__}: {e})")

    print(f"\n  {out_name} is a normal candidate name: view_image('{out_name}'), "
          f"measure('{out_name}'), pick_best('{out_name}').")
    print(f"  Cost is an ESTIMATE (~{EST_CENTS:.0f}c) - gpt-image-2 bills by "
          f"token, not per image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
