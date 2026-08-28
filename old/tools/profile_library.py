#!/usr/bin/env python3
"""Turn a folder of finished PDP assets into a layout profile.

    python tools/profile_library.py --library ../Garment_Library

For each category folder it measures every asset and writes
profiles/<category>.json - the framing convention, expressed as numbers.

Two things make this work across garments of different shapes:

  * It picks the FRAMING AXIS from the data. Leggings are scaled to a target
    height and their width falls out of the silhouette (measured: height
    84.8-87.9%, width 40.3-54.0%). Bras are the opposite - width 78.5-80.9%,
    height 46.5-82.4%. Constraining the tight axis and leaving the other free is
    what lets one profile cover skinny and wide-leg, bralette and cami.

  * Every tolerance comes from the library, not from a guess. A hardcoded
    3-degree tilt bar rejected 2 of 12 real, shipped bras - because racerback
    straps pull the principal axis off square. The library defines what
    acceptable looks like; this reads it off.

The profile constrains PLACEMENT only. Silhouette is always judged against the
product's own photo, so a profile that is slightly wrong makes a garment sit a
little large or small in frame - never a different shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import common as C

EXTS = {".psd", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def measure(path: Path) -> dict:
    m, _ = C.garment_mask(path)
    H, W = m.shape
    ys, xs = np.nonzero(m)
    if not len(xs):
        raise ValueError("no garment found")
    s = C.shape_stats(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    # Plate tone from a border frame, away from the garment.
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    b = 12
    edge = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                           a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])

    return {
        "file": path.name,
        "height_pct": (y1 - y0) / H * 100,
        "width_pct": (x1 - x0) / W * 100,
        "margin_top": y0 / H * 100,
        "margin_bottom": (1 - y1 / H) * 100,
        "margin_left": x0 / W * 100,
        "margin_right": (1 - x1 / W) * 100,
        "centre_x": s["cx"],
        "centre_y": s["cy"],
        "tilt": s["tilt"],
        "symmetry": s["symmetry"],
        "aspect": (x1 - x0) / max(1, y1 - y0),
        "plate_rgb": [round(float(v), 1) for v in edge.mean(axis=0)],
        "canvas": list(Image.open(path).size),
    }


def mad_outliers(v: np.ndarray, k: float = 4.0, floor: float = 2.0) -> np.ndarray:
    """Median-absolute-deviation outliers, with an absolute floor.

    The floor matters. Bra widths span 78.5-80.9% around a median of 79.7, so
    the MAD is ~0.15 and a point 1.2 away scores 5 robust sigmas - flagging two
    perfectly normal, shipped assets. A distribution this tight has no room for
    a purely relative test. Something is only an outlier if it is BOTH far in
    sigmas and far in percentage points.
    """
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    dev = np.abs(v - med)
    if mad == 0:
        return dev > floor
    return (dev / (1.4826 * mad) > k) & (dev > floor)


def build(cat_dir: Path, k: float, floor: float) -> dict | None:
    files = sorted(p for p in cat_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        return None
    rows, failed = [], []
    for p in files:
        try:
            rows.append(measure(p))
        except Exception as e:
            failed.append(f"{p.name}: {e}")
    if not rows:
        return None

    h = np.array([r["height_pct"] for r in rows])
    w = np.array([r["width_pct"] for r in rows])

    # The framing axis is whichever the studio actually held constant.
    axis = "height" if (h.std() / h.mean()) <= (w.std() / w.mean()) else "width"
    prim = h if axis == "height" else w

    out = mad_outliers(prim, k, floor)
    for r, o in zip(rows, out):
        r["outlier"] = bool(o)
    kept = [r for r, o in zip(rows, out) if not o]
    p = prim[~out]

    def band(key):
        v = np.array([r[key] for r in kept])
        return {"median": round(float(np.median(v)), 2),
                "min": round(float(v.min()), 2),
                "max": round(float(v.max()), 2)}

    prof = {
        "category": cat_dir.name.lower(),
        "n_assets": len(rows),
        "n_used": len(kept),
        "canvas": kept[0]["canvas"],
        "framing_axis": axis,
        "framing_target_pct": round(float(np.median(p)), 2),
        "framing_tolerance_pct": round(float(max(np.median(p) - p.min(),
                                                 p.max() - np.median(p))), 2),
        "free_axis": "width" if axis == "height" else "height",
        "free_axis_observed": band("width_pct" if axis == "height" else "height_pct"),
        "centre_x": band("centre_x"),
        "centre_y": band("centre_y"),
        "margins": {k2: band(f"margin_{k2}")
                    for k2 in ("top", "bottom", "left", "right")},
        "tilt_tolerance_deg": round(float(np.abs([r["tilt"] for r in kept]).max()), 2),
        "symmetry_min": round(float(min(r["symmetry"] for r in kept)), 3),
        "plate_rgb": [round(float(v), 1) for v in
                      np.median([r["plate_rgb"] for r in kept], axis=0)],
        "outliers": [r["file"] for r, o in zip(rows, out) if o],
        "unreadable": failed,
        "assets": rows,
    }
    return prof


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", type=Path, default=C.ROOT / "Garment_Library")
    ap.add_argument("--out", type=Path, default=C.ROOT / "profiles")
    ap.add_argument("--mad", type=float, default=4.0,
                    help="outlier cutoff in robust sigmas on the framing axis")
    ap.add_argument("--floor", type=float, default=2.0,
                    help="an asset must also be this many percentage points "
                         "from the median to count as an outlier")
    a = ap.parse_args()

    if not a.library.is_dir():
        return print(f"Not a directory: {a.library}") or 1
    a.out.mkdir(parents=True, exist_ok=True)

    for cat in sorted(d for d in a.library.iterdir() if d.is_dir()):
        prof = build(cat, a.mad, a.floor)
        if not prof:
            print(f"{cat.name}: no readable assets, skipped")
            continue
        dst = a.out / f"{prof['category']}.json"
        dst.write_text(json.dumps(prof, indent=2))
        t = prof["framing_target_pct"]
        print(f"{prof['category']:10} n={prof['n_used']}/{prof['n_assets']}  "
              f"fit {prof['framing_axis']} to {t}% "
              f"(+-{prof['framing_tolerance_pct']})  "
              f"{prof['free_axis']} free "
              f"{prof['free_axis_observed']['min']}-"
              f"{prof['free_axis_observed']['max']}%  "
              f"tilt <={prof['tilt_tolerance_deg']} deg  "
              f"plate {prof['plate_rgb']}")
        if prof["outliers"]:
            print(f"           excluded: {', '.join(prof['outliers'])}")
        if prof["unreadable"]:
            print(f"           unreadable: {'; '.join(prof['unreadable'])}")
        print(f"           -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
