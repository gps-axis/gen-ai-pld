#!/usr/bin/env python3
"""Step 4 - measure every candidate against the SOURCE. Nothing billed.

    python tools/measure.py --run runs/<stamp>

Everything here asks one question: does this still look like your product?

Framing is deliberately not graded. Position, scale, tilt and margins are a
transform on a layer and the retouching team fixes them in seconds; the
deliverable is a cutout, not a composited plate. Grading framing was actively
harmful - the candidates rejected for it carried the BEST colour fidelity and
texture in every batch, because they were the ones that left the product alone.

The single framing fault still checked is clipping, because pixels that were
never captured cannot be retouched back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C

CANVAS = (1024, 768)   # H, W - common canvas so every mask is comparable
RES_ORDER = {"1K": 1, "2K": 2, "4K": 3}


def auto_select(arch: Path):
    """Every generated image at the highest resolution present.

    A probe generated at the same resolution as the final batch is a candidate -
    same model, same prompt, same pixels. Excluding it by filename discarded the
    three best images of a real run.
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
    print(f"measuring {len(paths)} image(s) at {res}"
          + (f"; {dropped} lower-resolution probe(s) held back" if dropped else ""))
    return paths, f"images at {res}"


def profile(path: Path) -> dict:
    """Shape statistics on a common canvas, plus dimensions at native scale."""
    m, _ = C.garment_mask(path)
    native = dict(C.rigid_dims(m), clip=C.clipped(m))
    # Normalise the rigid dimensions by the mask's own frame, so a 2K probe and
    # a 4K candidate are directly comparable.
    H, W = m.shape
    native["length"] /= H
    native["top_width"] /= W
    big = C.resize_mask(m, CANVAS)
    return {**C.shape_stats(big), **native, "mask": big}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--off-set", type=Path, default=C.INPUTS / "off_set_image.jpg")
    ap.add_argument("--reference", type=Path, default=None,
                    help="optional. Only prints an informational IoU; never "
                         "gates anything. Framing is the retoucher's job.")
    ap.add_argument("--pattern", default=None,
                    help="glob to measure in archive/. Default is automatic: "
                         "every generated image at the highest resolution "
                         "present, cand_ or probe_ alike.")
    ap.add_argument("--reject-colour", type=float, default=20.0,
                    help="garment colour drift from the source. Measured: the "
                         "source against itself 0.4, a desaturated candidate 56.")
    ap.add_argument("--lay-target", type=float, default=None,
                    help="target solidity - how much of its own convex hull the "
                         "silhouette fills. Rises as splayed limbs close, so it "
                         "is the one number that measures whether the re-lay "
                         "actually happened. Defaults to the reference's, then "
                         "the category profile's.")
    ap.add_argument("--reject-specks", type=int, default=50,
                    help="background blobs above which the plate is not a "
                         "plate. Clean runs measure 0-1; a run whose prompt "
                         "asked for transparency painted a checkerboard and "
                         "measured 1587-3135, which made every other number "
                         "meaningless.")
    ap.add_argument("--reject-shape", type=float, default=8.0,
                    help="percent that the garment's LENGTH or TOP-BAND WIDTH "
                         "may differ from the source. These survive a "
                         "legitimate re-lay; overall width does not, since "
                         "closing splayed legs narrows it on purpose.")
    a = ap.parse_args()

    arch = a.run / "archive"
    try:
        manifest = json.loads((arch / "seeds.json").read_text())
    except Exception:
        manifest = {}
    manifest = {k: v for k, v in manifest.items() if isinstance(v, dict)}
    if a.pattern:
        cands, label = sorted(arch.glob(a.pattern)), a.pattern
    else:
        cands, label = auto_select(arch)
    if not cands:
        return print(f"No {label} in {arch}. Run generate.py first.") or 1

    src = profile(a.off_set)
    src_rgb = C.garment_rgb(a.off_set, src["mask"])
    src_seam = C.seam_energy(a.off_set, src["mask"])
    ref = profile(a.reference) if a.reference and a.reference.exists() else None
    lay_target = a.lay_target if a.lay_target is not None else (
        ref["solidity"] if ref else None)

    print(f"source     length {src['length']*100:.1f}% of frame   "
          f"top-band {src['top_width']*100:.1f}%   solidity {src['solidity']:.3f}   "
          f"symmetry {src['symmetry']:.3f}   seam {src_seam:.2f}")
    if ref:
        print(f"reference  solidity {ref['solidity']:.3f}   "
              f"symmetry {ref['symmetry']:.3f}   (informational only)")
    print()

    rows = []
    for p in cands:
        d = profile(p)
        rgb = C.garment_rgb(p, d["mask"])
        r = {
            "cand": p.stem,
            "file": str(p),
            "colour_drift": float(np.linalg.norm(rgb - src_rgb)),
            "len_pct": (d["length"] / src["length"] - 1) * 100 if src["length"] else 0.0,
            "topw_pct": (d["top_width"] / src["top_width"] - 1) * 100
                        if src["top_width"] else 0.0,
            "solidity": d["solidity"],
            "symmetry": d["symmetry"],
            "seam_vs_src": float(C.seam_energy(p, d["mask"]) / src_seam)
                           if src_seam else 0.0,
            "specks": C.speck_count(p),
            "clip": d["clip"],
        }
        if ref:
            r["iou_ref"] = C.iou(d["mask"], ref["mask"])
        r["prompt"] = manifest.get(p.stem, {}).get("prompt", "-")
        rows.append(r)

    # Ranked on product fidelity. Symmetry is the lay-quality term - it is the
    # one thing the model is genuinely being asked to improve.
    # A candidate whose solidity still matches the source has not re-laid
    # anything - it preserved the product by leaving it alone. That scores
    # perfectly on every fidelity term, so without this it wins outright.
    for r in rows:
        r["lay_gap"] = (abs(r["solidity"] - lay_target)
                        if lay_target is not None else 0.0)
        r["noop"] = abs(r["solidity"] - src["solidity"]) < 0.01
        r["score"] = (100
                      - r["colour_drift"] * 2.0
                      - abs(r["len_pct"]) * 1.5
                      - abs(r["topw_pct"]) * 1.5
                      - abs(r["seam_vs_src"] - 1) * 20
                      + r["symmetry"] * 20
                      - r["lay_gap"] * 150
                      - r["specks"] * 0.3)
        why = []
        if r["colour_drift"] > a.reject_colour:
            why.append("colour")
        if abs(r["len_pct"]) > a.reject_shape:
            why.append("length")
        if abs(r["topw_pct"]) > a.reject_shape:
            why.append("topwidth")
        if r["clip"]:
            why.append(f"clipped({r['clip']})")
        if r["specks"] > a.reject_specks:
            why.append(f"plate({r['specks']} specks)")
        r["reject"] = bool(why)
        r["reject_why"] = ",".join(why)

    rows.sort(key=lambda r: (r["reject"], -r["score"]))

    ioucol = f"{'IoUref':>7}" if ref else ""
    if lay_target is not None:
        print(f"lay target solidity {lay_target:.3f}   source {src['solidity']:.3f} "
              f"(a candidate still at the source's value did not re-lay)\n")
    prompts = {r["prompt"] for r in rows}
    pcol = f"{'prompt':>9}" if len(prompts) > 1 else ""
    if len(prompts) > 1:
        print(f"{len(prompts)} different prompts in this folder - the column "
              f"shows which produced each image.\n")
    print(f"{'cand':12} {'colour':>7} {'len%':>7} {'topw%':>7} {'sym':>6} "
          f"{'solid':>6} {'seam/src':>9} {'spk':>4}{ioucol}{pcol} {'score':>7}")
    for r in rows:
        flag = (f"  REJECT {r['reject_why']}" if r["reject"]
                else "  NO-OP, garment not re-laid" if r["noop"] else "")
        iou = f"{r['iou_ref']:7.3f}" if ref else ""
        pc = f"{r['prompt']:>9}" if len(prompts) > 1 else ""
        print(f"{r['cand']:12} {r['colour_drift']:7.1f} {r['len_pct']:+7.1f} "
              f"{r['topw_pct']:+7.1f} {r['symmetry']:6.3f} {r['solidity']:6.3f} "
              f"{r['seam_vs_src']:9.2f} {r['specks']:4d}{iou}{pc} "
              f"{r['score']:7.1f}{flag}")

    keep = [r for r in rows if not r["reject"]]
    print(f"\n{len(rows) - len(keep)} rejected, {len(keep)} remain.")
    print(f"KEEP  {','.join(r['cand'] for r in keep)}"
          f"{'  <- only these are eligible' if keep else '  (nothing survived)'}")
    noops = [r["cand"] for r in keep if r["noop"]]
    if noops:
        print(f"NO-OP (kept the source lay, did not close/square the garment): "
              f"{', '.join(noops)}")
    print("Framing is NOT graded - the deliverable is a cutout and the retoucher "
          "places it. Only clipping is fatal.")
    print("len%/topw% are vs the source; near 0 means the product's real "
          "dimensions survived. seam/src near 1.00 is the source's own detail.")
    print("Numbers narrow the field. LOOK at the top candidates before picking.")

    out = {"source": {k: src[k] for k in
                      ("length", "top_width", "solidity", "symmetry")},
           "source_seam": src_seam,
           "reject_colour": a.reject_colour, "reject_shape": a.reject_shape,
           "candidates": [{k: v for k, v in r.items() if k != "mask"} for r in rows]}
    stem = ("metrics" if not a.pattern
            else a.pattern.replace("*", "").replace(".png", "").strip("_") or "metrics")
    (arch / f"{stem}.json").write_text(json.dumps(out, indent=2))
    # Contact sheets, always. Wrinkles have no metric, so the sheet is the only
    # cheap way to judge them - one comparative vision call over several
    # candidates instead of an absolute verdict on each.
    try:
        import contact
        sheets = contact.main_for(a.run)
        if sheets:
            print(f"\ncontact sheets: {', '.join(s.name for s in sheets)}  "
                  f"<- LOOK at these for wrinkles; nothing above measures them")
    except Exception as e:
        print(f"  (contact sheets not built: {e})")

    C.log(a.run, f"measured {len(rows)} ({label}), {len(rows)-len(keep)} rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
