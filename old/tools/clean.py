#!/usr/bin/env python3
"""Automatic pre-clean: erase tags and pins, drop the background, plate white.

    python tools/clean.py --run runs/<stamp>

This runs before any re-lay, and it exists to give the generative pass less to
do. Every job the re-lay model is asked to perform is a chance for it to drift
the colour or invent a seam, and background removal is not a job it should have:
a segmentation model does it without touching a single garment pixel.

Two steps, deliberately in this order:

  1. `fal-ai/image-editing/object-removal` - erases the hang tag, price ticket
     and other paperwork by text prompt. GENERATIVE, so it runs FIRST, while the
     real surrounding fabric is still there to reconstruct from. Skipped unless
     --remove is given.

     **--remove names PAPER, never hardware.** It used to end in "safety pins,
     clips", and on a pair of navy leggings the eraser read the two reflective
     thigh-pocket zips as clips and painted them out - 4.11% of the frame
     spliced back, colour drift 0.8, so nothing warned. describe.py then
     inventoried the CLEANED image, wrote "Two drop-in pockets" and listed
     "Zippered Pockets" under NOT PRESENT, generate.py appended that to all ten
     prompts, and the stage 3 construction gate compares candidates against this
     file - so every check downstream agreed the zips had never existed. Four
     zipper-less flats shipped. Two pin heads left at a waistband corner are a
     retouch afterthought; an erased zip is a different product.

     The default used to end in "hanger" as well, and it cost the same way. On
     an olive crop top shot flat - no hanger anywhere in the frame - the eraser
     read the folded strap tops as one and painted both out: 370 px off the top
     of a 3277 px garment, 4.98% of the frame spliced back where the good run
     spliced 3.40%, colour drift 3.0 so the drift warning (>8) stayed quiet.
     birefnet then correctly matted away fabric that was no longer there. Two
     runs of the same photo minutes apart, one intact and one with the straps
     cut short, because the eraser is generative and re-rolls. **Name only what
     is actually in the photograph, and only if it is paper.**

     Even the trimmed default is not safe on its own. On the very first run
     after "hanger" came out, `pins, tags` made the eraser read the woven
     starburst logo on the chest as a pin and paint it out - repainted to
     fabric, in the middle of a flat panel, so the outline never moved and no
     geometric check could have seen it. The vision check below caught it at 98%
     confidence and the erase-off retry brought the logo back. "pins" stays in
     the default because the task asks for it, but it is on probation, and it is
     the reason the vision check is load-bearing rather than decorative.

  2. `fal-ai/birefnet/v2` - alpha matte, composited onto white. Pure
     segmentation: it decides which pixels are garment, it does not repaint
     them. Re-running the crop-top photo through matting alone reproduced the
     source outline exactly, which is how step 1 was identified as the culprit.

Three checks now stand between those calls and the file everything downstream
trusts, in increasing order of cost and decreasing order of certainty:

  * **The splice guard** (`splice`) refuses, per blob, any erase that turns
    garment into plate or plate into a new dark object. The eraser may repaint
    fabric; it may not delete fabric. Free, deterministic, and the reason the
    strap failure cannot reach the output any more.
  * **The outline gate** (`outline_report`) compares the finished silhouette's
    bounding box against the source's. This catches loss at the garment's edge
    whatever produced it, including the matte.
  * **The vision check** (`vision_verdict`) asks the model to compare source and
    result and name any construction that went missing. This is the only one of
    the three that can see an erased zip in the MIDDLE of a panel, where the
    outline never moves. It is also the noisiest, so it can ask for the one
    retry but it cannot fail the run on its own.

The retry, when either of the first two fires, is not a re-roll of the same
stochastic call - that is a coin flip charged twice. It is the deterministic
fallback: clean again with the generative erase switched off, which is known to
preserve the outline. One retry, then the run fails loudly rather than handing
a mutilated plate to ten generations.

The result is written as `archive/offset_upload.jpg`, which is what the re-lay
prompt then sees. Garment colour is measured before and after and reported, so a
generative step that shifted the product is visible rather than silent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import common as C

ERASER = "fal-ai/image-editing/object-removal"
MATTE = "fal-ai/birefnet/v2"

# How far an outline edge may move inwards, as a fraction of the garment's own
# extent on that axis. Measured on the run this gate was written for: a good
# clean moved the top edge 0.8%, the bad one 12.6%. A 2% line sits in the middle
# of a 16x gap, so it is not a number that needs tuning.
EDGE_TOL = 0.02
# Area is the backstop, not the test. Matting legitimately shaves ~5% off by
# trimming anti-aliased edge pixels, and the bad run only lost 7.6%, so this
# threshold catches catastrophe and leaves the discrimination to the bbox.
AREA_TOL = 0.10


def garment_colour(path: Path, cue: str | None = None) -> np.ndarray:
    """Mean garment colour. `cue` is the SOURCE's cue, and every call after the
    first passes it.

    Drift is a before-and-after number, so both measurements have to find the
    garment the same way or the drift reports the mask moving rather than the
    colour changing. That is not theoretical: a near-black garment measured
    [61,57,52] on the source and [210,200,179] on the cleaned copy, a drift of
    242.8, because the second measurement had landed on a beige strip at the
    edge of the frame.
    """
    m, _ = C.garment_mask(path, cue=cue)
    return C.garment_rgb(path, m)


def splice(full: Image.Image, erased: Image.Image, base: np.ndarray | None = None,
           plate: float | None = None, min_frac=2e-5, feather=6.0,
           bad_frac=0.02, bad_abs=1e-4,
           margin: float = C.PLATE_MARGIN) -> tuple[Image.Image, float, list[dict]]:
    """Put ONLY the erased patch back into the full-resolution original.

    The eraser silently returns a much smaller image - measured 3072x4096 in,
    880x1184 out, a 3.5x loss. Accepting that would hand the re-lay model a
    soft, quarter-size photograph, which is the opposite of preserving the
    product.

    So the erase runs small, and the result is used only where it actually
    changed something. Everything else stays the original's own pixels at full
    resolution. The changed region is found by differencing, keeping blobs big
    enough to be a tag rather than resampling noise, and feathering the seam.

    `base` (the source garment mask) and `plate` (the backdrop level) turn on the
    guard, and every caller should pass them. Each candidate blob is classified
    before it is accepted:

      deleted - it was fabric inside the outline and is now plate. This is the
                strap failure, and it is the one thing a *removal* step must
                never do.
      invented - it was plate outside the outline and is now dark. The mirror
                fault: the eraser adding an object instead of taking one away.

    Either one above both a relative and an absolute floor drops that blob, and
    those pixels keep the photograph's own. Note what the guard deliberately
    permits: a tag lying ON the garment erases to fabric, which is nowhere near
    plate level, and a tag lying on the backdrop was never inside the outline.
    Erasing paperwork is untouched; only eating the product is blocked.

    Returns (image, patched %, rejected blobs).
    """
    from scipy import ndimage
    up = erased.resize(full.size, Image.LANCZOS)
    a = np.asarray(full, dtype=np.float32)
    b = np.asarray(up, dtype=np.float32)

    d = ndimage.gaussian_filter(np.abs(a - b).mean(axis=2), 3.0)
    changed = d > 18.0
    lab, n = ndimage.label(changed)
    total = changed.size

    deletes = invents = None
    if base is not None and plate is not None and base.any():
        big = np.asarray(Image.fromarray((base * 255).astype(np.uint8))
                         .resize(full.size, Image.NEAREST)) > 127
        # The mask is measured on a 1024px grid and blown back up, so its edge
        # is only accurate to a few full-resolution pixels. Erode and dilate by
        # that slop before asking either question, so ordinary anti-aliasing at
        # the outline is never mistaken for the eraser eating the garment.
        scale = max(1, round(full.size[1] / base.shape[0]))
        slop = max(4, 2 * scale)
        inside = ndimage.binary_erosion(big, np.ones((slop, slop)))
        outside = ~ndimage.binary_dilation(big, np.ones((slop, slop)))
        # The SAME margin the mask was built with. A guard that draws the line
        # between fabric and plate somewhere else from the outline it is
        # policing is guarding a different garment - which is the whole reason
        # PLATE_MARGIN was given a name in the first place.
        was_fabric = a.mean(axis=2) < plate - margin
        now_plate = b.mean(axis=2) > plate - margin
        deletes = inside & was_fabric & now_plate
        invents = outside & ~was_fabric & ~now_plate

    keep = np.zeros_like(changed)
    rejects: list[dict] = []
    if n:
        boxes = ndimage.find_objects(lab)
        sizes = ndimage.sum(changed, lab, range(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz < min_frac * total:              # tag-sized, not noise
                continue
            sl = boxes[i - 1]
            blob = lab[sl] == i
            if deletes is not None:
                nd = int((blob & deletes[sl]).sum())
                ni = int((blob & invents[sl]).sum())
                bad, kind = (nd, "deleted garment") if nd >= ni else \
                            (ni, "invented an object")
                if bad > bad_frac * sz and bad > bad_abs * total:
                    rejects.append({
                        "kind": kind,
                        "blob_pct": round(100.0 * sz / total, 3),
                        "bad_pct": round(100.0 * bad / total, 3),
                        "rows": [int(sl[0].start), int(sl[0].stop)],
                        "cols": [int(sl[1].start), int(sl[1].stop)]})
                    continue
            keep[sl] |= blob
    if not keep.any():
        return full, 0.0, rejects

    keep = ndimage.binary_dilation(keep, np.ones((25, 25)))
    m = ndimage.gaussian_filter(keep.astype(np.float32), feather)[..., None]
    out = a * (1 - m) + b * m
    return Image.fromarray(out.astype(np.uint8)), float(keep.mean() * 100), rejects


def outline_report(before: dict, after: dict, edge_tol=EDGE_TOL,
                   area_tol=AREA_TOL) -> tuple[list[str], list[str]]:
    """Compare two silhouettes. Returns (failures, measurements).

    Every edge is measured and printed whether or not it fails, because the
    number is the evidence: "top edge held to 0.8%" is what makes the next
    person believe the gate rather than switch it off.
    """
    if not after["bbox"]:
        return ["no garment found in the cleaned image at all"], []
    if not before["bbox"]:
        return [], ["no garment found in the source; outline gate skipped"]

    bt, bb, bl, br = before["bbox"]
    at, ab, al, ar = after["bbox"]
    h, w = max(bb - bt, 1), max(br - bl, 1)
    fails, notes = [], []
    for name, moved, ext, axis in (("top", at - bt, h, "height"),
                                   ("bottom", bb - ab, h, "height"),
                                   ("left", al - bl, w, "width"),
                                   ("right", br - ar, w, "width")):
        frac = moved / ext                       # positive = pulled inwards
        notes.append(f"{name:6s} edge {frac * 100:+6.1f}% of garment {axis} "
                     f"({moved:+d} px on {ext})")
        if frac > edge_tol:
            fails.append(f"{name} edge pulled in {frac * 100:.1f}% of the "
                         f"garment {axis} (limit {edge_tol * 100:g}%)")
    lost = (before["area"] - after["area"]) / max(before["area"], 1e-9)
    notes.append(f"area   {-lost * 100:+6.1f}%")
    if lost > area_tol:
        fails.append(f"outline lost {lost * 100:.1f}% of its area "
                     f"(limit {area_tol * 100:g}%)")
    elif lost < -0.15:
        # The outline GREW. Nothing the clean does can add garment, so this is
        # the source silhouette having been incomplete - which happens when part
        # of the garment shares a tone with the background it was shot against.
        # runs/20260820_115631: a black-and-cream bra on a cream wall, where the
        # cream side panels were invisible until the plate went white and the
        # outline grew 30.4%. Not a failure - the gate exists to catch loss - but
        # it means the source outline understated the garment, and the splice
        # guard was policing that same understated outline.
        notes.append(f"NOTE: the outline grew {-lost * 100:.1f}%. The clean "
                     f"cannot add garment, so the SOURCE outline was "
                     f"incomplete - part of the garment reads as background in "
                     f"the original photograph. The gate still holds; the "
                     f"splice guard was working from that partial outline.")
    return fails, notes


VISION_PROMPT = """Image 1 is the ORIGINAL photograph of a garment, laid flat.
Image 2 is the same photograph after an automatic clean-up step.

The clean-up was SUPPOSED to do exactly these things. None of them is a fault,
and you must not report any of them:
  - remove the background and replace it with plain white
  - remove paper hang tags, price tickets, barcodes, pins and clips
  - change the overall brightness or colour slightly
  - soften the shadows

You are checking ONE thing: whether any part of THE GARMENT ITSELF is missing,
cut short, or altered in image 2. Straps and strap tips, a shoulder, a hem, a
corner, a sleeve, a waistband, a pocket, a zip, a drawcord, an elastic, a logo,
a seam, a mesh panel - anything that is construction rather than paperwork.

Work through the garment part by part. Pay particular attention to thin parts at
the edge of the frame and to hardware in the middle of a flat panel: those are
the two places an automatic eraser takes something and leaves a plausible
picture behind. If the garment has something in image 1 that is absent or
truncated in image 2, name it.

Return ONE JSON object, nothing else:
{"garment_intact": true or false,
 "missing": "<the part that is gone or cut short, or 'none'>",
 "confidence": <0-100>}"""


def _thumb(src: Path, dst: Path, max_dim: int = 1024) -> Path:
    """Downscale with PIL rather than vision.ensure_small, which shells out to
    `sips`. This step runs inside the Linux container too."""
    im = Image.open(src).convert("RGB")
    im.thumbnail((max_dim, max_dim), Image.LANCZOS)
    im.save(dst, quality=88)
    return dst


def vision_verdict(before: Path, after: Path, base_url: str, model: str,
                   timeout: int) -> dict | None:
    """Ask the model to name garment parts that went missing. None on any
    failure - an unreachable server must not stop a clean that measured fine."""
    try:
        import vision as V
        client = V.Client(base_url or V.DEFAULT_BASE_URL,
                          model or V.DEFAULT_MODEL, timeout)
        client.resolve_model()
        content = [V.text_part("Image 1 - ORIGINAL:"), V.image_part(before),
                   V.text_part("Image 2 - CLEANED:"), V.image_part(after),
                   V.text_part(VISION_PROMPT)]
        v = V.parse_json_blob(client.chat(content, max_tokens=300,
                                          temperature=0.0))
        v["model"] = client.model
        return v
    except Exception as e:                        # noqa: BLE001 - advisory only
        print(f"  vision check unavailable ({type(e).__name__}: {e}); "
              f"the outline gate above still stands.")
        return None


def clean_once(staged: Path, out: Path, arch: Path, remove: str, icc,
               base: np.ndarray, plate: float, before: np.ndarray,
               fal_client, requests, cue: str | None = None,
               margin: float = C.PLATE_MARGIN) -> dict:
    """One full pass: optional erase, then matte onto white. Writes `out`.

    `cue` is how the garment was found in the SOURCE, and everything measured
    here is measured the same way - see garment_colour().
    """
    work = Image.open(staged).convert("RGB")
    cur, info = staged, {"patch": 0.0, "rejects": [], "erase_drift": None}

    if remove.strip():
        print(f"erasing       '{remove}' via {ERASER}")
        r = fal_client.subscribe(ERASER, arguments={
            "image_url": fal_client.upload_file(str(cur)),
            "prompt": remove, "output_format": "png"}, with_logs=False)
        items = r.get("images", [])
        if not items:
            raise RuntimeError("object-removal returned no image")
        img = Image.open(requests.get(items[0]["url"], stream=True,
                                      timeout=300).raw).convert("RGB")
        spliced, patch, rejects = splice(work, img, base=base, plate=plate,
                                         margin=margin)
        cur = arch / "_clean_erased.png"
        spliced.save(cur)
        after = garment_colour(cur, cue)
        drift = float(np.linalg.norm(after - before))
        info.update(patch=patch, rejects=rejects, erase_drift=drift)
        print(f"              model returned {img.width}x{img.height}; spliced "
              f"{patch:.2f}% of the frame back into the full-resolution "
              f"original")
        print(f"              -> {spliced.width}x{spliced.height}  "
              f"garment RGB {after.round(0).tolist()}  drift {drift:.1f}")
        for rj in rejects:
            print(f"  GUARD: refused an erase that {rj['kind']} - blob "
                  f"{rj['blob_pct']:.2f}% of the frame, {rj['bad_pct']:.2f}% of "
                  f"it product, rows {rj['rows'][0]}-{rj['rows'][1]} cols "
                  f"{rj['cols'][0]}-{rj['cols'][1]}. Those pixels keep the "
                  f"photograph's own.")
        if patch == 0.0 and not rejects:
            print("  NOTE: nothing changed enough to splice - no tag found, or "
                  "the eraser did nothing.")
        if drift > 8.0:
            print(f"  WARNING: the erase moved the garment colour by {drift:.1f}. "
                  f"It is a generative step; check the result before trusting it.")

    print(f"matting       {MATTE}")
    r = fal_client.subscribe(MATTE, arguments={
        "image_url": fal_client.upload_file(str(cur)),
        "output_format": "png", "refine_foreground": True,
        "operating_resolution": "2048x2048"}, with_logs=False)
    url = (r.get("image") or {}).get("url") or r.get("images", [{}])[0].get("url")
    if not url:
        raise RuntimeError(f"birefnet returned no image: {list(r)}")
    cut = Image.open(requests.get(url, stream=True, timeout=300).raw).convert("RGBA")

    # Composite onto white. The re-lay model gets a clean plate, and every
    # garment pixel is still the photograph's own - matting chooses pixels, it
    # does not repaint them.
    plate_img = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(plate_img, cut).convert("RGB")
    flat.save(out, quality=95, subsampling=0, icc_profile=icc) if icc else \
        flat.save(out, quality=95, subsampling=0)

    final = garment_colour(out, cue)
    info["drift"] = float(np.linalg.norm(final - before))
    info["size"] = (flat.width, flat.height)
    print(f"              -> {flat.width}x{flat.height}  "
          f"garment RGB {final.round(0).tolist()}  drift from source "
          f"{info['drift']:.1f}")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--off-set", type=Path, default=C.INPUTS / "off_set_image.jpg")
    ap.add_argument("--out", type=Path,
                    help="default <run>/archive/offset_upload.jpg")
    ap.add_argument("--remove", default="pins, tags",
                    help="objects to erase by name. Paper only, and only what "
                         "is actually in the photograph - naming hardware or a "
                         "hanger that is not there costs product. Pass '' to "
                         "skip the generative erase and only drop the "
                         "background.")
    ap.add_argument("--long-side", type=int, default=4096,
                    help="long side of the image that gets uploaded")
    ap.add_argument("--keep-steps", action="store_true",
                    help="also write the intermediate erased image")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="times to re-clean with the generative erase switched "
                         "off after a failed check. The retry is deterministic, "
                         "so more than one buys nothing.")
    ap.add_argument("--edge-tol", type=float, default=EDGE_TOL,
                    help="how far an outline edge may move inwards, as a "
                         "fraction of the garment's extent")
    ap.add_argument("--area-tol", type=float, default=AREA_TOL,
                    help="how much outline area may disappear, as a fraction")
    ap.add_argument("--no-guard", action="store_true",
                    help="do not police what the eraser changes (diagnostic)")
    ap.add_argument("--no-vision-check", action="store_true",
                    help="skip the model-side check for missing construction")
    ap.add_argument("--base-url", default="", help="vision server, default the "
                                                   "one the harness uses")
    ap.add_argument("--model", default="", help="vision model, default the "
                                                "first the server lists")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()

    if not a.off_set.exists():
        return print(f"Not found: {a.off_set}") or 1
    run = a.run or C.session_run_dir()
    arch = run / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    out = a.out or arch / "offset_upload.jpg"

    C.fix_ca_bundle()
    C.load_fal_key()
    import fal_client
    import requests

    src = Image.open(a.off_set)
    icc = src.info.get("icc_profile")
    work = src.convert("RGB")
    work.thumbnail((a.long_side, a.long_side), Image.LANCZOS)
    staged = arch / "_clean_input.jpg"
    work.save(staged, quality=95, subsampling=0)

    # How the garment is found in the SOURCE decides how it is found in
    # everything measured against the source. Deciding that per image instead
    # let a near-black garment be located by brightness in one and by colour in
    # the other, and the gate reported the disagreement as 81.3% of the garment
    # destroyed.
    src_ev = C.garment_evidence(staged)
    cue = src_ev["cue"]
    before = C.garment_rgb(staged, src_ev["mask"])
    base = src_ev["mask"]
    plate = C.plate_level(staged)
    src_sil = C.silhouette(staged, cue=cue)
    print(f"source        {work.width}x{work.height}  "
          f"garment RGB {before.round(0).tolist()}")
    print(f"              plate level {plate:.0f}, garment {src_sil['area']*100:.1f}% "
          f"of frame, outline bbox {src_sil['bbox']}")
    print(f"              found by {cue} ({src_ev['cue_why']}), fabric/plate line "
          f"at {src_ev['luma_margin']:.0f} below plate; everything below is "
          f"measured the same way")
    if a.no_guard:
        print("              GUARD OFF - the eraser may change the outline")

    remove = a.remove
    verdict, notes, fails, info = None, [], [], {}
    reason = ""
    audit = {"source": src_sil, "attempts": []}

    for attempt in range(a.max_retries + 1):
        if attempt:
            remove = ""
            print(f"\nretry {attempt}/{a.max_retries}: cleaning again with the "
                  f"generative erase switched OFF. Matting alone cannot repaint "
                  f"a pixel, so this is the deterministic fallback, not a "
                  f"re-roll of the same call.")
            C.log(run, f"pre-clean retry {attempt}: erase disabled after "
                       f"{reason}")
        try:
            info = clean_once(staged, out, arch, remove, icc, None if a.no_guard
                              else base, plate, before, fal_client, requests,
                              cue=cue, margin=src_ev["luma_margin"])
        except RuntimeError as e:
            print(e)
            return 1

        out_sil = C.silhouette(out, cue=cue)
        fails, notes = outline_report(src_sil, out_sil, a.edge_tol, a.area_tol)
        print("\noutline gate  (cleaned vs source, + means pulled inwards)")
        for line in notes:
            print(f"              {line}")

        verdict = None
        if not fails and not a.no_vision_check:
            bpath = _thumb(staged, arch / "_clean_vcheck_before.jpg")
            apath = _thumb(out, arch / "_clean_vcheck_after.jpg")
            verdict = vision_verdict(bpath, apath, a.base_url, a.model, a.timeout)
            if verdict is not None:
                ok = bool(verdict.get("garment_intact", True))
                miss = str(verdict.get("missing", "none")).strip()
                print(f"vision check  {verdict.get('model', '?')}: "
                      f"garment_intact={ok} missing={miss!r} "
                      f"confidence={verdict.get('confidence', '?')}")

        audit["attempts"].append({
            "remove": remove, "erase_patch_pct": info.get("patch"),
            "guard_rejected": info.get("rejects"), "drift": info.get("drift"),
            "outline": out_sil, "outline_fails": fails, "vision": verdict})

        vision_bad = bool(verdict) and not verdict.get("garment_intact", True) \
            and str(verdict.get("missing", "none")).strip().lower() not in ("", "none")
        if not fails and not vision_bad:
            break
        if fails:
            print("\n  OUTLINE GATE FAILED:")
            for f in fails:
                print(f"    - {f}")
            reason = fails[0]
        if vision_bad:
            print(f"\n  VISION CHECK: the model says the clean lost "
                  f"{verdict.get('missing')!r}.")
            reason = reason or f"vision found {verdict.get('missing')!r} missing"
        if attempt >= a.max_retries:
            break
        if not remove.strip():
            print("  the erase was already off; a further retry would change "
                  "nothing.")
            break

    (arch / "clean_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")

    if not a.keep_steps:
        for p in (arch / "_clean_input.jpg", arch / "_clean_erased.png",
                  arch / "_clean_vcheck_before.jpg",
                  arch / "_clean_vcheck_after.jpg"):
            if p.exists() and p != out:
                p.unlink()

    rejected = sum(len(x["guard_rejected"] or []) for x in audit["attempts"])
    tail = f", guard refused {rejected} erase{'s' if rejected != 1 else ''}" \
        if rejected else ""
    if fails:
        print("\n" + "=" * 74)
        print("PRE-CLEAN FAILED - the cleaned image is not the same garment.")
        for f in fails:
            print(f"  {f}")
        print(f"  audit: {arch / 'clean_audit.json'}")
        print("  The file was written anyway so it can be looked at, but do NOT")
        print("  generate from it: every candidate would inherit the loss and")
        print("  every downstream check would agree the garment always looked")
        print("  like this.")
        print("=" * 74)
        C.log(run, f"PRE-CLEAN FAILED outline gate: {fails[0]}{tail}")
        return 1

    note = ""
    if len(audit["attempts"]) > 1 and not remove.strip():
        # The erase was dropped and the result then came back clean. That is a
        # caught defect, not a quiet success, and steps.log is where anyone
        # reading the run afterwards finds out the source was nearly damaged.
        note = (f"; ERASE DROPPED after {reason} - this image is matted only, "
                f"so nothing generative touched it")
        print(f"\n  The generative erase was dropped after {reason}, and the "
              f"matted-only result checks out. Paperwork the eraser would have "
              f"removed is still in the photograph; that is the cheaper "
              f"mistake.")
    elif verdict is not None and not verdict.get("garment_intact", True):
        note = (f"; vision still flags {verdict.get('missing')!r} with the "
                f"erase off - likely a false positive, but worth a look")
        print(f"\n  NOTE: the vision check flags {verdict.get('missing')!r}, but "
              f"the erase is off and the outline held, so nothing repainted a "
              f"pixel. Shipping, flagged.")

    print("This is what the re-lay prompt will see. It no longer needs to "
          "remove a background or a tag - only to re-lay and de-wrinkle.")
    C.log(run, f"pre-cleaned onto white, drift {info.get('drift', 0.0):.1f}, "
               f"outline held{tail} (2 calls, unpriced){note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
