#!/usr/bin/env python3
"""Three free numbers about a candidate, measured against the original source.

    python tools/metrics.py --source archive/source_clean.jpg \
                            --candidate archive/cand_03.png

    python tools/metrics.py --run runs/<stamp>            # every candidate

No model, no network, no money. Every term is a comparison with the SOURCE, so
the question being asked is "is this still the same garment, laid flatter" and
never "is this a nice photo".

WHAT THIS IS FOR. The generator has two ways to satisfy a re-lay prompt: lay the
real garment flat, or draw a new garment that is already flat. The second one
looks better. It is the single failure that eyes reliably miss, and it is the
only reason this file exists - a redraw moves the outline and repaints the
texture, and both show up here before anyone has to notice them by looking.

WHAT THIS IS NOT. Not a grade, not a gate, not a ranking. Nothing in the harness
blocks, sorts or rejects on these numbers. The previous pipeline turned exactly
these measurements into a weighted score with a pass mark and spent whole runs
arbitrating between the number and the picture; the number won often enough to
ship redraws. They are advice, printed next to the image, and the model's eyes
have the last word in both directions.

ALWAYS AGAINST THE ORIGINAL. When a candidate is generated from another
candidate, the comparison is still to `source_clean.jpg`, never to the parent.
Comparing to the parent measures one hop and calls a three-hop drift small.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402


def compare(source: Path, candidate: Path, long_side: int = 1024) -> dict:
    """The three numbers, plus the cheap extras, for one candidate.

    The source is read first so its detection cue can be forced onto the
    candidate. common.garment_evidence picks between a chroma cue and a
    luminance cue by whichever finds more garment; left to decide freely, two
    images of the same garment can be measured by different cues, and the
    resulting IoU compares one definition of "garment" against another.
    """
    src = C.garment_evidence(Path(source), long_side=long_side)
    cue = src["cue"]
    cand = C.garment_evidence(Path(candidate), long_side=long_side, cue=cue)

    # A forced cue can find nothing at all, and an empty mask does not read as a
    # failure - it reads as IoU 0.000, dE 91.5, wrinkle x0.00, which looks like a
    # measurement of a catastrophe rather than the absence of one. Measured: the
    # greyscale reference against a colour source returns exactly that, because
    # the source's chroma cue has no chroma to find in a desaturated image.
    #
    # So when the forced cue comes back near-empty, fall back to letting the
    # candidate choose its own cue, and say which one was used. A cross-cue
    # comparison is weaker than a matched one - that is why the cue is forced in
    # the first place - but it is a number, and a stated caveat beats a confident
    # zero.
    cue_note = ""
    if float(cand["mask"].mean()) < C.CHROMA_MIN_AREA:
        free = C.garment_evidence(Path(candidate), long_side=long_side)
        if float(free["mask"].mean()) >= C.CHROMA_MIN_AREA:
            cue_note = (f"source cue '{cue}' found nothing here; measured with "
                        f"'{free['cue']}' instead, so these are cross-cue and "
                        f"weaker than usual")
            cand, cue = free, f"{cue}->{free['cue']}"
        else:
            cue_note = ("no garment found in the candidate under either cue - "
                        "these numbers are meaningless, look at the image")

    # Is the candidate's mask even plausible? Everything below is a comparison of
    # two masks, so a mask that did not find the garment does not produce a bad
    # score - it produces a confident number about nothing, and a run quoted
    # "IoU 0.325" in its own shipping rationale on exactly that basis.
    #
    # The failure is specific and common in this catalogue: a CREAM garment on a
    # WHITE plate. Chroma has almost no colour to separate and luminance has
    # almost no contrast, so both cues collapse. Measured on
    # runs/20260827_220727/archive/cand_05.png - a sweater filling most of the
    # frame - the mask came back at 4.1% of it.
    #
    # The same check catches the opposite failure for free: a candidate carrying
    # a SECOND garment covers far more of the frame than the source does.
    src_area, cand_area = float(src["mask"].mean()), float(cand["mask"].mean())
    if not cue_note and src_area > 0:
        ratio = cand_area / src_area
        if ratio < 0.5:
            cue_note = (f"the garment mask covers {cand_area*100:.1f}% of this "
                        f"frame against the source's {src_area*100:.1f}% - the "
                        f"outline was probably not found (a pale garment on a "
                        f"pale plate defeats both cues). TREAT EVERY NUMBER "
                        f"BELOW AS UNRELIABLE and judge by eye.")
        elif ratio > 1.8:
            cue_note = (f"the garment mask covers {cand_area*100:.1f}% of this "
                        f"frame against the source's {src_area*100:.1f}%, which "
                        f"is far too much for one garment. Look for a SECOND "
                        f"garment in the picture, or a plate dark enough to be "
                        f"counted as fabric. The numbers below are unreliable "
                        f"either way.")

    src_mask, cand_mask = src["mask"], cand["mask"]

    iou = C.silhouette_iou(src_mask, cand_mask)
    drift = C.colour_drift(C.garment_rgb(Path(source), src_mask),
                           C.garment_rgb(Path(candidate), cand_mask))

    w_src = C.wrinkle_energy(Path(source), src_mask, long_side=long_side)
    w_cand = C.wrinkle_energy(Path(candidate), cand_mask, long_side=long_side)
    ratio = (w_cand / w_src) if w_src > 0 else float("nan")

    return {
        "source": str(source),
        "candidate": str(candidate),
        "cue": cue,
        "silhouette_iou": round(float(iou), 4),
        "colour_de": round(float(drift["de"]), 2),
        "wrinkle_source": round(float(w_src), 4),
        "wrinkle_candidate": round(float(w_cand), 4),
        "wrinkle_ratio": (None if w_src <= 0 else round(float(ratio), 3)),
        "specks": int(C.speck_count(Path(candidate))),
        "clipped": C.clipped(cand_mask),
        "plate_level": round(float(C.plate_level(Path(candidate))), 4),
        "area_pct": round(float(cand_mask.mean()) * 100, 1),
        "cue_note": cue_note,
    }


def line(m: dict, name: str | None = None) -> str:
    """One-line rendering - this is what gets printed under a candidate image."""
    who = name or Path(m["candidate"]).stem
    ratio = "n/a" if m["wrinkle_ratio"] is None else f"{m['wrinkle_ratio']:.2f}"
    out = (f"{who}  IoU {m['silhouette_iou']:.3f}  "
           f"dE {m['colour_de']:.1f}  "
           f"wrinkle x{ratio}")
    if m["specks"]:
        out += f"  specks {m['specks']}"
    if m["clipped"]:
        out += f"  CLIPPED {m['clipped']}"
    if m.get("cue_note"):
        out += f"\n    ! {m['cue_note']}"
    return out


def candidates(arch: Path) -> list[Path]:
    """Every generated image in a run's archive, in the order they were made.

    Includes the segmented forms (cand_03s.png) - they are first-class names
    everywhere else, so they are measurable here too.
    """
    return sorted(p for p in arch.glob("cand_*.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path,
                    help="measure every cand_*.png in this run's archive")
    ap.add_argument("--source", type=Path,
                    help="defaults to <run>/archive/source_clean.jpg")
    ap.add_argument("--candidate", type=Path, action="append", dest="candidates",
                    help="repeatable; omit to take the whole archive")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--json", action="store_true", help="machine-readable")
    a = ap.parse_args()

    if a.run:
        arch = a.run / "archive"
        source = a.source or arch / "source_clean.jpg"
        cands = a.candidates or candidates(arch)
    else:
        if not a.source or not a.candidates:
            ap.error("give --run, or both --source and --candidate")
        source, cands = a.source, a.candidates

    if not Path(source).exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 1
    if not cands:
        print("no candidates to measure.", file=sys.stderr)
        return 1

    rows, failed = [], 0
    for c in cands:
        if not Path(c).exists():
            print(f"  {Path(c).name}: not found", file=sys.stderr)
            failed += 1
            continue
        try:
            rows.append(compare(Path(source), Path(c), a.long_side))
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the rest
            print(f"  {Path(c).name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"source {Path(source).name}"
              + (f"  (wrinkle energy {rows[0]['wrinkle_source']:.3f}, "
                 f"cue {rows[0]['cue']})" if rows else ""))
        for m in rows:
            print("  " + line(m))
    return 0 if rows and not failed else (0 if rows else 1)


if __name__ == "__main__":
    raise SystemExit(main())
