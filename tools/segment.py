#!/usr/bin/env python3
"""Cut the garment out of the off-set photo and stand it on a white plate.

    python tools/segment.py --run runs/<stamp>

Writes `archive/offset_upload.jpg`, the image the rest of the pipeline works
from. Step 0a runs this before the reference is matched, so the matcher scores
the same picture everything downstream uses.

This replaces the fal pre-clean, which called
`fal-ai/image-editing/object-removal` and has answered 403 "Access ... has been
restricted" since 2026-08-27. That failure did more damage than losing a clean:
prepare.py read the same flag for "should we clean" and "did cleaning work", so
a dead endpoint also silently switched off the construction inventory and every
prompt went out unanchored. clean.py is still here for when access returns.

WHAT THIS DOES AND DOES NOT DO. It drops the background. It does NOT remove a
hang tag, a price ticket, pins or clips - only segmentation is available on this
service, not object removal. Anything still attached to the garment survives
into the upload, which is why describe.py lists those items under TO REMOVE and
generate.py asks for them in the prompt. Background is solved here; attached
props are solved in words.

The one thing it does repair is the bite a clip hanger leaves: the segmenter
cuts the clip out with a square of waistband, and the generator draws a tab
where the notch is. Those notches are closed with fabric from either side
before the file is written - see common.close_top_bites().
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import common as C

URL = C.conf("SEGMENT_URL", "http://10.11.245.145:4000/sam3-segment")

# The endpoint above is the SAM3 service reached through the LiteLLM proxy,
# which is the same segmenter the direct address served - the proxy reports
# http://10.11.245.41:8780/segment as its upstream, and answers with the same
# JPEG. What changed is that the proxy authenticates: without a key it is a 401,
# so a run against it silently continued from the raw photo, background and all.
# Empty is still valid, for talking to a service that wants no auth.
API_KEY = C.conf("SEGMENT_API_KEY")

# A segmentation that ate the garment is worse than no segmentation: it is a
# white rectangle that every downstream measurement will happily describe. The
# guard is deliberately loose - it is catching catastrophe, not judging quality,
# and a real garment on a real plate covers far more than 4% of the frame.
MIN_AREA = 0.04


def post(src: Path, dst: Path, url: str = URL, timeout: int = 180) -> None:
    """POST the image as multipart/form-data and write the reply to `dst`.

    Hand-rolled rather than pulled from `requests`, which this project does
    depend on, because the body is one file field and building it here keeps
    the failure surface to one function.
    """
    boundary = f"----segment{uuid.uuid4().hex}"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; '
        + f'filename="{src.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        src.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if not data:
        raise RuntimeError("segmentation returned an empty body")
    dst.write_bytes(data)


def check(src: Path, out: Path) -> tuple[bool, str]:
    """Did a garment survive, and is it still the same one?

    Two failures worth catching separately. An empty result is the service
    working and finding nothing. A result that keeps only part of the garment -
    one sleeve, the body without the cuffs - still measures as a garment, and
    the only thing that reveals it is comparing the area against the source.
    """
    try:
        m, _ = C.garment_mask(out)
    except Exception as e:  # noqa: BLE001
        return False, f"no garment found in the result ({type(e).__name__})"
    area = float(m.mean())
    if area < MIN_AREA:
        return False, f"the result is effectively empty ({area*100:.1f}% ink)"
    try:
        src_area = float(C.garment_mask(src)[0].mean())
    except Exception:  # noqa: BLE001 - the source is the thing we cannot check
        return True, f"kept {area*100:.0f}% of the frame"
    # The source mask is measured against a cloth backdrop and picks up shadow,
    # so it reads LARGER than the true garment - a segmented result coming in
    # under it is normal and is not evidence of loss. Only a big shortfall is.
    if src_area > 0 and area < src_area * 0.55:
        return False, (f"the result keeps {area*100:.0f}% of the frame against "
                       f"the source's {src_area*100:.0f}% - most of the garment "
                       f"is missing")
    return True, f"kept {area*100:.0f}% of the frame (source {src_area*100:.0f}%)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--off-set", type=Path, default=C.INPUTS / "off_set_image.jpg")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--long-side", type=int, default=4096,
                    help="long side of the written upload")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()

    run = a.run or C.session_run_dir()
    if not a.off_set.exists():
        return print(f"Not found: {a.off_set}") or 1
    out = a.out or run / "archive" / "offset_upload.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"segmenting   {a.off_set.name} via {a.url}")
    raw = out.with_suffix(".seg_raw.jpg")
    try:
        post(a.off_set, raw, a.url, a.timeout)
    except Exception as e:  # noqa: BLE001 - every reachability failure reads alike
        raw.unlink(missing_ok=True)
        # An auth refusal is worth naming. It looks identical to "the box is
        # down" in the line above, and the fix is the opposite one.
        hint = ("Set SEGMENT_URL or --url if the service moved."
                if getattr(e, "code", None) not in (401, 403) else
                "The service refused the key: set SEGMENT_API_KEY to a key that "
                "is valid for this endpoint.")
        print(f"  segmentation FAILED: {type(e).__name__}: {e}\n"
              f"  the run continues from the raw photo, which still has its "
              f"background. {hint}")
        return 1

    ok, why = check(a.off_set, raw)
    if not ok:
        # Kept on disk deliberately. A result rejected sight-unseen is the one
        # thing nobody can debug, and it costs one file.
        bad = out.with_suffix(".seg_rejected.jpg")
        raw.replace(bad)
        print(f"  segmentation REJECTED: {why}\n"
              f"  written to {bad.name} so it can be looked at; the run "
              f"continues from the raw photo.")
        return 1

    im = Image.open(raw).convert("RGB")
    if max(im.size) > a.long_side:
        im.thumbnail((a.long_side, a.long_side), Image.LANCZOS)
    bites = []
    try:
        arr, bites = C.close_top_bites(np.asarray(im))
        if bites:
            im = Image.fromarray(arr)
    except Exception as e:  # noqa: BLE001 - a repair must never cost the cutout
        print(f"             bite repair skipped: {type(e).__name__}: {e}")
    im.save(out, quality=95, subsampling=0)
    raw.unlink(missing_ok=True)
    print(f"             -> {out.name}  {im.width}x{im.height}  "
          f"{out.stat().st_size/1e6:.1f} MB  ({why})")
    if bites:
        sizes = ", ".join(f"{b['width']}x{b['depth']}px" for b in bites)
        print(f"             closed {len(bites)} clip bite(s) in the top edge "
              f"({sizes}) with fabric from either side, so the generator "
              f"sees an unbroken waistband instead of a notch to copy.")
    print("             background dropped to white. A hang tag, ticket, pin "
          "or clip ON the garment is NOT removed by this step - look at the "
          "result, and ask for anything you can still see in the prompt.")
    C.log(run, f"segmented {a.off_set.name}, background dropped"
               + (f", {len(bites)} clip bite(s) closed" if bites else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
