#!/usr/bin/env python3
"""Contact sheets of every candidate, sized so wrinkles survive the vision call.

    python tools/contact.py --run runs/<stamp>

Judging wrinkles one image at a time costs a vision call each and asks the model
an absolute question it is bad at ("is this acceptable?"). A sheet asks the
comparative question it is good at ("which of these is smoothest?") in one call.

Two things make the sheet actually usable:

  * **Cells are cropped to the garment.** Better than half of a laydown frame is
    empty plate; tiling whole frames spends the budget on white.
  * **Few cells per sheet.** `view_image` downscales anything over 1024px, so a
    3x3 grid of leggings arrives at ~340px a cell and every crease is gone. At
    three across, each garment lands around 340x800 after the downscale, which
    is enough to see fold lines. More candidates make more sheets, not smaller
    cells.

The reference is prepended to every sheet as the leftmost cell, so "smoother
than the reference" is answerable without a second call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

import common as C

RES_ORDER = {"1K": 1, "2K": 2, "4K": 3}


def candidates(arch: Path, pattern: str | None):
    if pattern:
        return sorted(arch.glob(pattern))
    try:
        man = {k: v for k, v in json.loads((arch / "seeds.json").read_text()).items()
               if isinstance(v, dict)}
    except Exception:
        man = {}
    if not man:
        return sorted(arch.glob("cand_*.png"))
    best = max(RES_ORDER.get(v.get("resolution"), 0) for v in man.values())
    names = [k for k, v in man.items()
             if RES_ORDER.get(v.get("resolution"), 0) == best]
    return sorted(p for p in (arch / f"{n}.png" for n in names) if p.exists())


def garment_crop(path: Path, pad_frac: float = 0.02,
                 like: Path | None = None) -> Image.Image:
    """The garment's own bounding box, with a little air around it.

    `like` is the cleaned source. Passing it turns on the sanity check in
    common.garment_box: a box that fills a wildly different share of its frame,
    or has a wildly different aspect, is not used. Cells built without that
    check are what turned a real sheet into five band close-ups at five
    different magnifications, which is worse than useless - the sheet exists to
    make candidates comparable, and every cell claimed to show a garment.
    """
    g = C.garment_box(path, like=like)
    if not g["ok"]:
        print(f"  {path.name}: {g['note']}")
    im = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = g["box"]
    pad = int(pad_frac * im.height)
    return im.crop((max(0, x0 - pad), max(0, y0 - pad),
                    min(im.width, x1 + pad), min(im.height, y1 + pad)))


def build(cells, labels, out: Path, width: int) -> Image.Image:
    """One row, every cell at the SAME scale.

    Scaling each cell to a common width made tall and short garments arrive at
    different magnifications, so "which is smoothest" compared creases at
    different sizes. Solving for a shared height instead fills the row exactly
    and keeps every garment comparable - within a sheet and across sheets.
    """
    bar = 26
    gap = 8
    avail = width - gap * (len(cells) + 1)
    ratio = sum(im.width / im.height for im in cells)
    H_cell = max(1, int(avail / ratio))
    scaled = [im.resize((max(1, int(im.width * H_cell / im.height)), H_cell),
                        Image.LANCZOS) for im in cells]
    H = H_cell + bar + gap * 2
    sheet = Image.new("RGB", (width, H), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    x = gap
    for im, lab in zip(scaled, labels):
        sheet.paste(im, (x, gap))
        d.rectangle([x, gap, x + im.width - 1, gap + im.height - 1],
                    outline=(200, 200, 200))
        d.text((x + 3, gap + im.height + 6), lab, fill=(20, 20, 20))
        x += im.width + gap
    sheet.save(out, quality=94)
    return sheet


def main_for(run: Path, per_sheet: int = 3, width: int = 1024,
             pattern: str | None = None, reference: Path | None = None):
    """Build the sheets and return their paths. Used by measure.py."""
    arch = run / "archive"
    cands = candidates(arch, pattern)
    if not cands:
        return []
    ref = reference or arch / "offset_upload.jpg"
    ref_cell = [garment_crop(ref)] if ref.exists() else []
    ref_lab = ["SOURCE"] if ref_cell else []
    like = ref if ref.exists() else None
    import math
    n_sheets = max(1, math.ceil(len(cands) / per_sheet))
    base, extra = divmod(len(cands), n_sheets)
    out, i = [], 0
    for k in range(n_sheets):
        size = base + (1 if k < extra else 0)
        batch = cands[i:i + size]
        i += size
        dst = arch / f"sheet_{k + 1}.jpg"
        build(ref_cell + [garment_crop(p, like=like) for p in batch],
              ref_lab + [p.stem for p in batch], dst, width)
        out.append(dst)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="prepended to every sheet. Defaults to the run's own "
                         "cleaned source, which is the right thing to compare "
                         "wrinkles against.")
    ap.add_argument("--pattern")
    ap.add_argument("--per-sheet", type=int, default=3,
                    help="candidates per sheet, excluding the reference cell. "
                         "Above 3 the cells stop surviving the 1024px downscale "
                         "a vision call applies.")
    ap.add_argument("--width", type=int, default=1024,
                    help="sheet width. 1024 is the vision tool's ceiling; "
                         "larger just gets resampled back down.")
    a = ap.parse_args()

    arch = a.run / "archive"
    cands = candidates(arch, a.pattern)
    if not cands:
        return print(f"No candidates in {arch}.") or 1

    ref = a.reference or arch / "offset_upload.jpg"
    ref_cell = [garment_crop(ref)] if ref.exists() else []
    ref_lab = ["SOURCE"] if ref_cell else []
    like = ref if ref.exists() else None
    if like is None:
        print("  no cleaned source to check the crops against; every box below "
              "is taken on trust.")

    # Spread evenly across sheets. Fixed-size chunks left a remainder sheet
    # holding a single candidate, blown up to a different scale from every other
    # cell - so 10 at 3-per becomes 3,3,2,2 rather than 3,3,3,1.
    import math
    n_sheets = max(1, math.ceil(len(cands) / a.per_sheet))
    base, extra = divmod(len(cands), n_sheets)
    groups, i = [], 0
    for k in range(n_sheets):
        size = base + (1 if k < extra else 0)
        groups.append(cands[i:i + size])
        i += size

    made = []
    for n, batch in enumerate(groups, 1):
        cells = ref_cell + [garment_crop(p, like=like) for p in batch]
        labels = ref_lab + [p.stem for p in batch]
        out = arch / f"sheet_{n}.jpg"
        s = build(cells, labels, out, a.width)
        made.append(out)
        print(f"{out.name}  {s.width}x{s.height}  "
              f"{', '.join(p.stem for p in batch)}")

    print(f"\n{len(made)} sheet(s) over {len(cands)} candidate(s). Each cell is "
          f"cropped to the garment and sized to survive a vision call.")
    print("Ask which cell is smoothest - a comparison, not a verdict on one.")
    C.log(a.run, f"built {len(made)} contact sheet(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
