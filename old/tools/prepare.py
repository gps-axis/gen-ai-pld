#!/usr/bin/env python3
"""Step 1 - check the inputs and make the upload copy. Writes NO prompt.

    python tools/prepare.py

The prompt is the agent's job, deliberately. This script used to carry a
hardcoded prompt plus a lookup table asserting what each garment type looks
like - so the wording was fixed at authoring time by someone who had not seen
the images. That table described a bralette's neckline and straps while the
actual input was a pair of leggings.

It now does only what is mechanical: confirm both inputs, report what they are,
write the downscaled copy that gets uploaded, and leave a brief of the clauses
the prompt must cover. The agent looks at the two images, then writes
archive/prompt.txt itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

import common as C

BRIEF = """\
# Prompt brief

Look at these two, with `compare_images`, BEFORE writing anything:

    {run}/archive/offset_upload.jpg     <- the product, CLEANED
    {ref_path}                          <- the lay reference

**Use the upload, never `inputs/off_set_image.jpg`.** The upload is the only
image the model receives: tag erased, background dropped, plate white. The raw
input still has the hang tag and a real-world background, and a run that
described it asked for the tag to "stay in place" - all four candidates then
grew a tag that was not in the image sent.

Then write `{run}/archive/prompt.txt` with a bash heredoc, covering every clause
below in your own words, describing what you actually saw.

Do not copy this file. It is a checklist, not a prompt.

## Ask for ONE thing: re-lay the garment. Change nothing else.

The deliverable is a cutout on a transparent background - the retouch team sets
placement, scale, canvas and plate afterwards. **Do not ask for centring,
margins, scale or framing.** Nothing measures them, and every clause you spend
on them is a clause the model spends repainting a product that was already
right.

## Must cover

1. **Which image is which.** Image 1 is the product photo, image 2 the lay
   reference. The model reads them in that order.
2. **Everything visual comes from image 1** - colourway, fabric, texture, print,
   seams, topstitching, hardware. Invent nothing not visible in image 1.

   **The construction inventory already exists** at
   `archive/garment_description.md`, and `generate.py` appends it to every
   prompt automatically. When `inputs/Design_BOM.png` is present it is
   transcribed from that spec sheet and is authoritative; otherwise it is
   inferred from the photo and only its NOT-PRESENT half is sent, because the
   positive half fabricates.

   **Do not describe seams yourself.** Two different vision models, asked what
   this garment has, both invented a seam running down each leg - the spec
   sheet says "No inseam". Your prompt needs one line saying the construction is
   reproduced exactly as specified and nothing is added; the inventory does the
   rest.

   **If you do describe construction yourself, be specific.** A generic "keep the seams" is weak;
   walk the garment and list what you can actually see, with where it sits and
   how it is stitched - the waistband join and whether it is topstitched, each
   panel line, pocket openings and their edges, the centre-front and inseam
   seams, the coverstitch at each hem, any elastic edge, gusset or logo. Say
   they must stay **sharp, continuous, and the same colour and stitch type as
   in image 1** - tonal thread stays tonal, and no seam may be smoothed away,
   softened, doubled or moved. This is the clause that decides whether the
   product survives, and it is the first thing lost when a model is told to
   smooth a garment.
3. **Image 2 shows only how the garment should be ARRANGED.** {ref_note} It is
   not a shape, colour, fabric or framing reference. If it is built differently
   from image 1 - a different number of straps, closures or panels - ignore all
   of that. The output has exactly the parts visible in image 1, no more and no
   fewer.

{bleed_note}
4. **The lay itself - this is the actual job.** Name what is untidy in image 1
   and what square looks like for THIS garment: legs parallel and closed rather
   than splayed, straps flat and symmetric, hems level, no twists or folds.
5. **Flatness.** It stays laid flat as in image 1. No volume, body, 3D shaping,
   draping or a worn look. The model adds volume unless told not to.
6. **Relax the creases - do not iron the fabric out of existence.** Ask for
   this, in your own words: **relax the handling folds and creases so the
   fabric lies flat; keep the knit's real texture - do not iron it into a
   smooth painted surface.**

   Say it that way round. The maximal version - "completely wrinkle-free, as if
   freshly steamed and pressed, no creases, no rumples, no fold lines" - is
   what a real prompt asked for, and it is this project's documented driver of
   redraws: told to remove every trace of texture, the model repaints the
   panel. The grade now measures the consequence directly. `wrink` scores the
   DISTANCE between the candidate's surface texture and the source's, in both
   directions, so a candidate ironed smoother than the real garment loses
   exactly as much as a creased one. `generate.py` warns when it sees the
   absolutes.

   Be precise about what stays, or the model smooths the product away with the
   creases: **seams, topstitching, panel lines, pockets, elastic edges, the
   waistband join and the fabric's own knit or weave all remain, sharp.** Only
   the temporary creases from handling go.
7. **Proportions unchanged.** The garment keeps image 1's real dimensions - the
   same length, the same waistband or band width. Closing the lay must not
   stretch or slim the product.
8. **Background: it is already clean. Keep it.** Image 1 has been pre-cleaned
   before you ever see it - a segmentation model dropped the real-world
   background and plated it pure white, and an eraser removed the hang tag and
   any pins. There is nothing left to remove.

   So ask only that the background **stays** a plain, even, seamless WHITE
   studio plate, and that nothing is added to it.

   **Never ask for a transparent or removed background.** The model cannot
   output an alpha channel, so it paints a transparency checkerboard into the
   pixels instead - a real run came back with a literal grey-and-white checker
   pattern, which counted as 1587-3135 background specks and drove every score
   below -400. Transparency is produced locally afterwards by `cutout.py`.
   `generate.py` refuses a prompt that asks for one.

   **Do not ask for the tag to be removed either** - it is already gone, and
   naming it invites the model to invent one to erase.
9. **Nothing cropped.** The whole garment stays inside the frame with clear space
   on every side. This is the one framing fault that matters: a clipped hem or
   strap tip cannot be retouched back.
10. **Square, as its own clause.** Do not leave this tangled into the wrinkle
    sentence - it is a different instruction and it gets lost. Say: **lay the
    garment straight and square: band parallel to the bottom of the frame,
    straps arranged symmetrically, no rotation or tilt, the whole garment
    inside the frame with white space around it.**

    That is about the GARMENT'S OWN alignment. Do not extend it into centring,
    margins or scale - placement is deliberately ungraded here, the retouch team
    sets it, and clauses about framing make the model repaint a product that was
    already right. The `sym` term rewards a square lay at 15% of the grade,
    which is as much weight as it should carry: a redrawn garment is always more
    symmetric than a real one, so symmetry cannot be allowed to outrank
    fidelity.
11. **The same face of the garment.** Never state or imply which side is showing
    - not "shown from the back", not "the front view", not "reverse side".
    `generate.py` REFUSES a prompt that names a viewpoint, because that is
    exactly how a garment comes back flipped: one run's prompt opened "the
    garment is shown from the back" and four of its ten candidates came back
    showing the other face. The clause that keeps the face correct is appended
    to every prompt automatically; yours does not need to mention sides at all.

## Inputs as measured

- off-set:    {off}
- reference:  {ref}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--off-set", type=Path, default=C.INPUTS / "off_set_image.jpg")
    ap.add_argument("--reference", type=Path, default=C.INPUTS / "reference_image.jpg")
    ap.add_argument("--upload-long-side", type=int, default=4096,
                    help="long side of the copy that gets uploaded")
    # Default OFF, matching harness.py: the pre-clean's object-removal endpoint
    # is restricted (403) and every attempt ends in a traceback. --clean puts it
    # back if the account regains access.
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    default=False,
                    help="skip the automatic tag/background pre-clean (default)")
    ap.add_argument("--clean", dest="clean", action="store_true",
                    help="re-enable the fal pre-clean")
    ap.add_argument("--out", type=Path, help="run folder (default runs/<stamp>)")
    a = ap.parse_args()

    for p in (a.off_set, a.reference):
        if not p.exists():
            return print(f"Not found: {p}") or 1

    # One folder per session. Calling this twice returns the SAME folder rather
    # than a fresh one, so the image budget cannot be reset by starting over.
    # --out is ignored while a session is active. A run spent three turns
    # creating its own folder, writing a prompt into it, being refused by
    # generate.py, and redoing the work in the right place.
    run = C.session_run_dir()
    if a.out and a.out.resolve() != run.resolve():
        import os
        if os.environ.get("LAYDOWN_SESSION"):
            print(f"Ignoring --out {a.out}: this session's folder is {run}. "
                  f"One run means one folder.")
        else:
            run = a.out
    existed = (run / "archive").exists()
    (run / "archive").mkdir(parents=True, exist_ok=True)
    (run / "output").mkdir(parents=True, exist_ok=True)
    if existed:
        n = len(list((run / "archive").glob("cand_*.png")))
        print(f"REUSING the existing run folder for this session "
              f"({n} image(s) already generated). Nothing was reset.")

    off, ref = Image.open(a.off_set), Image.open(a.reference)
    off_desc = f"{a.off_set.name} {off.width}x{off.height} mode={off.mode}"
    ref_desc = f"{a.reference.name} {ref.width}x{ref.height} mode={ref.mode}"
    print(f"off-set    {off_desc}")
    print(f"reference  {ref_desc}")
    bom = C.INPUTS / "Design_BOM.png"
    print(f"spec sheet {bom.name + ' ' + str(Image.open(bom).size) if bom.exists() else 'NONE - construction will be inferred from the photo, which fabricates'}")

    if ref.mode == "L":
        note = ("The reference is GREYSCALE - say so explicitly and tell the "
                "model to ignore its tone completely. Without that clause it "
                "reads the grey as a colour target and desaturates the "
                "garment.")
        print("  reference is greyscale - the prompt MUST tell the model to "
              "ignore its tone")
    else:
        note = ("The reference is in colour, so say plainly that the colour "
                "still comes from image 1 and none of it from image 2.")
        print("  NOTE: reference is not greyscale - colour may bleed from it")

    # The pre-clean - tag erased, background dropped, plate white - normally
    # happened HERE, on the agent's first turn. It now runs at step 0, before
    # the reference is matched, so the matcher scores the same image the rest of
    # the pipeline works from. Same work, same spend, earlier.
    #
    # So the job here is to verify, not to redo. Cleaning again would be two
    # billed calls for an image already on disk, and a second cleaning path that
    # can disagree with the first. clean.py is run below only when step 0 did
    # not leave one - a direct `prepare.py` outside the harness, or a step 0
    # that failed - which keeps this script usable on its own.
    # An upload already on disk is step 0a's segmentation, and it is kept
    # WHATEVER --clean says. That ordering matters: step 0 matched the reference
    # against this exact image, so replacing it with a raw copy here would leave
    # the reference chosen from one picture and the generation made from
    # another. The old code only kept it when `a.clean` was true, and a.clean is
    # by then a record of whether cleaning WORKED, not of what was asked for -
    # so a failed clean quietly threw away a good segmentation too.
    # Two different questions, and conflating them is what broke this before.
    # `have_clean` - is there a prepared upload on disk? Segmentation satisfies
    # it. `props_removed` - has anything ATTACHED to the garment been taken
    # out? Only clean.py's generative eraser ever did that, and it is the one
    # that answers 403. Segmentation drops the background and leaves the hang
    # tag, ticket and pins exactly where they were, so a segmented image is
    # clean to look at and still dirty to describe.
    up_path = run / "archive" / "offset_upload.jpg"
    have_clean = up_path.exists()
    props_removed = False
    if have_clean:
        w, h = Image.open(up_path).size
        print(f"pre-cleaned  archive/offset_upload.jpg {w}x{h}  "
              f"{up_path.stat().st_size/1e6:.1f} MB  (from step 0a, not redone)")
    elif a.clean:
        print("no pre-cleaned upload from step 0a; cleaning now.")
        import subprocess
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("clean.py")),
                            "--run", str(run), "--off-set", str(a.off_set),
                            "--long-side", str(a.upload_long_side)],
                           capture_output=True, text=True)
        print(r.stdout.rstrip() or r.stderr.rstrip())
        if r.returncode != 0 or not up_path.exists():
            print("  pre-clean failed; falling back to a plain downscaled copy.")
            a.clean = False
        else:
            have_clean = props_removed = True
    if not have_clean:
        up = off.convert("RGB")
        up.thumbnail((a.upload_long_side, a.upload_long_side), Image.LANCZOS)
        up.save(up_path, quality=95, subsampling=0,
                icc_profile=off.info.get("icc_profile"))
        print(f"upload copy  {up.width}x{up.height}  "
              f"{up_path.stat().st_size/1e6:.1f} MB  (NOT pre-cleaned)")

    # NO letterboxing here, and the reason is worth keeping.
    #
    # The stacked-sweater ghost looked like an aspect problem: the source is
    # landscape 4096x3072, every generation is 3:4 portrait, so the model was
    # handed a wide picture and asked for a tall one, and the leftover band was
    # where the second garment grew. Padding the source to 3:4 was the obvious
    # fix and it was WRONG - runs/20260827_104532 went from two ghosts in ten to
    # five or six. Padding leaves the garment filling 56% of the frame instead
    # of 100%, and more empty canvas gave the model MORE room to put a second
    # copy in, not less. C.pad_to_aspect is kept for callers that want it, but
    # the source is sent at its own shape.
    #
    # The ghost is not geometric. Across three runs no prompt has ever said how
    # many garments to draw; generate.py now says it outright.

    # Inventory the construction. generate.py appends this to every prompt, so
    # each draw is anchored to one written spec rather than to whatever the
    # re-lay model infers - which is where invented seams and lost topstitching
    # have come from.
    #
    # This used to be gated on `a.clean`, which is why runs/20260827_095355 has
    # no inventory at all: the pre-clean 403'd, the failure handler reassigned
    # a.clean from "should we clean" to "did cleaning work", and the gate read
    # the second meaning. Worse, the only warning about a missing inventory
    # lived INSIDE the block, so the one case that needed it printed nothing.
    #
    # The gate is gone. The inventory now runs from whatever image is actually
    # on disk, and it runs free on the local model, so there is no spend to
    # protect by skipping it. What the clean used to remove is instead recorded
    # by describe.py under TO REMOVE and asked for in the prompt.
    desc = run / "archive" / "garment_description.md"
    if not desc.exists():
        import subprocess
        cmd = [sys.executable, str(Path(__file__).with_name("describe.py")),
               "--run", str(run)]
        if not props_removed:
            # Keyed on props_removed, NOT on whether the image was cleaned up.
            # A segmented image looks clean - white plate, no room - while still
            # carrying every tag and pin that was pinned to the garment, and
            # that is exactly the case where describe.py needs telling. Left
            # unsaid it inventories them as construction, and a prompt written
            # from a raw input once asked for the tag to "stay in place" while
            # all four candidates kept it.
            cmd.append("--dirty-source")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.rstrip() or r.stderr.rstrip())
    if not desc.exists():
        # Outside the block on purpose. A missing inventory is the single
        # failure most likely to go unnoticed, so it warns whether the step was
        # skipped, crashed, or was never reached.
        print("  WARNING: no construction inventory; prompts will not carry "
              "one, and nothing downstream will anchor the construction.")

    # Step 0 measured what the reference carries beyond the lay. That is worth
    # a clause of the agent's own prompt as well as the one generate.py appends
    # automatically - the two say the same thing, and the failure they describe
    # cost four of ten images on a real run.
    bleed_note = ""
    try:
        sel = json.loads((run / "reference_selection.json").read_text())
        risk = sel.get("construction_risk") or {}
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        risk = {}
    if risk.get("flagged"):
        bleed_note = (
            f"   **Step 0 measured this reference as differing from the product "
            f"in: {', '.join(risk.get('terms', []))}.** It reported: "
            f"\"{risk.get('line', '')}\" None of that is on the garment in image "
            f"1. Say so explicitly in your own words - `generate.py` appends a "
            f"clause about it too, and this is the failure that put a neckline "
            f"seam and strap topstitching on four of ten candidates in a real "
            f"run.")
        print(f"  reference bleed risk: {', '.join(risk.get('terms', []))} - "
              f"the brief carries a clause about it")

    (run / "archive" / "prompt_brief.md").write_text(
        BRIEF.format(run=run, ref_note=note, off=off_desc, ref=ref_desc,
                    ref_path=a.reference, bleed_note=bleed_note))
    print("brief        archive/prompt_brief.md - 11 clauses the prompt must cover")
    print("NO prompt written. Look at both images, then write archive/prompt.txt.")

    w, h = Image.open(up_path).size
    # Reports what the image actually IS, in the two dimensions that differ.
    # It read `a.clean` before, which by this point means "did clean.py work",
    # so a perfectly good segmentation was logged as "(not cleaned)" - the same
    # conflation that cost the construction inventory. Both facts are recorded
    # because they are separately actionable: a raw background is a prompt
    # problem, attached props are a TO REMOVE problem.
    state = ("background dropped" if have_clean else "RAW - background included")
    if not props_removed:
        state += ", tag/pins not removed"
    C.log(run, f"prepared, upload {w}x{h} ({state})")
    print(f"\nRUN_DIR={run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
