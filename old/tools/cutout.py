#!/usr/bin/env python3
"""Turn a candidate into a transparent-background PNG for the retouch team.

    python tools/cutout.py --run runs/<stamp> --cand cand_01

The deliverable is the garment on its own layer. Placement, scale, canvas and
plate are then a transform the retoucher applies in seconds - which is why the
pipeline stopped grading framing at all.

Two things matter here and nothing else does:

  * The edge. A binary mask cuts a hard, aliased outline that reads as a
    paste-up. `soft_alpha()` ramps alpha across the plate-to-garment transition
    so the natural edge softness survives.
  * The pixels. RGB is copied through untouched - no colour management, no
    resampling, no sharpening. The only thing added is the alpha channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import common as C


def decontaminate(rgb: np.ndarray, alpha: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Remove the plate's contribution from partly transparent edge pixels.

    A semi-transparent edge pixel is a mix: alpha of garment over (1-alpha) of
    plate. Kept as-is it still carries the plate's colour, which is invisible
    against a light background and shows as a bright halo against a dark one.
    Solving C = a*F + (1-a)*P for F recovers the garment's own colour, so the
    layer composites cleanly onto anything.
    """
    a = alpha[..., None]
    safe = np.maximum(a, 0.25)     # dividing by a tiny alpha amplifies noise
    f = np.clip((rgb - (1.0 - a) * plate) / safe, 0, 255)
    # Never let the estimate move AWAY from the garment. On a light plate a
    # recovered pixel can only get darker; letting it brighten is what turned
    # edge pixels pure white and produced the halo this exists to remove.
    if float(plate.mean()) > 128:
        f = np.minimum(f, rgb)
    else:
        f = np.maximum(f, rgb)
    return np.where(a > 0.98, rgb, f)


def cut(src: Path, dst: Path, feather: float, trim: bool, pad: int,
        clean: bool = True) -> dict:
    alpha = C.soft_alpha(src, feather)
    im = Image.open(src).convert("RGB")
    if alpha.shape != (im.height, im.width):
        alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                           .resize(im.size, Image.LANCZOS), dtype=np.float32) / 255.0

    rgb = np.asarray(im, dtype=np.float32)
    if clean:
        b = 8
        edge = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                               rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
        rgb = decontaminate(rgb, alpha, np.median(edge, axis=0))

    out = Image.fromarray(np.dstack([rgb.astype(np.uint8),
                                     (alpha * 255).astype(np.uint8)]), "RGBA")

    ys, xs = np.nonzero(alpha > 0.02)
    if not len(xs):
        raise SystemExit(f"No garment found in {src}")
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(im.width, xs.max() + 1 + pad), min(im.height, ys.max() + 1 + pad))
    if trim:
        out = out.crop(box)

    out.save(dst)
    cov = float((alpha > 0.5).mean() * 100)
    return {"size": out.size, "coverage_pct": cov,
            "box": [int(v) for v in box],
            "mb": dst.stat().st_size / 1e6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--cand", help="stem in archive/, e.g. cand_01")
    ap.add_argument("-i", "--image", type=Path, help="any image, instead of --run/--cand")
    ap.add_argument("-o", "--out", type=Path, help="output path")
    ap.add_argument("--feather", type=float, default=0.0,
                    help="extra blur on alpha, in pixels. Default 0 - the "
                         "plate-to-garment luminance ramp is already soft, "
                         "and blurring puts alpha on pixels that hold no "
                         "garment, which is what created a white halo.")
    ap.add_argument("--trim", action="store_true",
                    help="crop to the garment plus --pad. Off by default: the "
                         "full frame keeps the layer registered to the original.")
    ap.add_argument("--pad", type=int, default=24,
                    help="pixels kept around the garment when trimming")
    ap.add_argument("--no-clean", dest="clean", action="store_false",
                    help="skip edge colour decontamination")
    a = ap.parse_args()

    src = a.image or (a.run / "archive" / f"{a.cand}.png"
                      if a.run and a.cand else None)
    if not src or not src.exists():
        return print(f"Not found: {src}. Pass --image, or --run with --cand.") or 1
    dst = a.out or (src.parent / f"{src.stem}_cutout.png")

    info = cut(src, dst, a.feather, a.trim, a.pad, a.clean)
    print(f"{dst.name}  {info['size'][0]}x{info['size'][1]}  "
          f"{info['mb']:.1f} MB  garment covers {info['coverage_pct']:.1f}% "
          f"of the saved frame")
    print("RGB untouched inside the garment; only edge pixels were "
          "decontaminated." if a.clean else
          "RGB untouched; only an alpha channel was added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
