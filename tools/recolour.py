#!/usr/bin/env python3
"""Put the real garment's colour back on a finished candidate. Free. MANUAL.

    python tools/recolour.py --run runs/<stamp> --candidate cand_10
    python tools/recolour.py --run runs/<stamp> --candidate cand_10p --no-whiten

Writes `cand_10c.png` beside the candidate (a `c` after any `s` or `p`), and
`archive/last_recolour.json` with what was measured and done. The candidate is
never touched.

NOT PART OF A RUN. This ran automatically on the winner for exactly one run,
20260902_104708, and the operator removed it from the process: the correction
aimed at the wrong statistic and shipped an olive garment for a cream one. The
target has since moved to the fully lit fabric and the result on that run's
winner measures dE 0.2, but the step stays out of the pipeline. It is here for
when someone wants to see a candidate in the source's colour by hand.

WHAT IT DOES. Reads the source garment's colour off its LIT fabric - the
brightest few percent of the garment by luminance, see common.TONE_LIT_PCT - and
scales the candidate's garment per channel in linear light until its lit fabric
matches, through a soft mask that leaves the plate alone. Then, unless
--no-whiten, takes the plate to white, since a repainted candidate arrives on a
plate a few levels grey. A change of colour is a change of albedo and albedo is
multiplicative, so every crease-to-highlight ratio the candidate had survives.

WHAT IT CANNOT DO. Restore texture, construction or a missing logo. It moves
colour and nothing else, and the pins-and-labels read on the candidate is
still the check for the rest.

Exit codes: 0 written   1 broke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import metrics as M  # noqa: E402

CAND_RE = re.compile(r"^cand_\d+[spc]*$")


def recoloured_name(candidate: str) -> str:
    return f"{candidate}c"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=None,
                    help="run folder (default: this session's)")
    ap.add_argument("--candidate", required=True,
                    help="cand_10, cand_10p, cand_10s ...")
    ap.add_argument("--source", type=Path, default=None,
                    help="the garment whose colour is wanted; default "
                         "<run>/archive/source_clean.jpg")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <run>/archive/<candidate>c.png")
    ap.add_argument("--no-whiten", action="store_true",
                    help="leave the plate at the level the candidate came with")
    a = ap.parse_args()

    run = (a.run or C.session_run_dir()).resolve()
    arch = run / "archive"
    name = a.candidate.strip()
    if not CAND_RE.match(name):
        print(f"not a candidate name: {name!r}", file=sys.stderr)
        return 1
    cand = arch / f"{name}.png"
    if not cand.exists():
        print(f"candidate not found: {cand}", file=sys.stderr)
        return 1
    source = (a.source or arch / "source_clean.jpg").resolve()
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 1
    out = (a.out or arch / f"{recoloured_name(name)}.png").resolve()

    colour = C.garment_colour(source)
    if not colour.get("found"):
        print(f"no garment found in {source.name} ({colour.get('why')}), so "
              f"there is no colour to restore", file=sys.stderr)
        return 1
    print(f"source lit fabric: RGB {[round(v) for v in colour['rgb']]}  "
          f"L* {colour['L']:.1f} a {colour['a']:.1f} b {colour['b']:.1f}")

    rec = C.recolour_garment(cand, out, colour["rgb"],
                             whiten_plate=not a.no_whiten)
    rec.update({"candidate": name, "child": out.stem, "source": str(source),
                "source_colour": colour})
    if rec.get("applied"):
        b, af = rec["rgb_before"], rec["rgb_after"]
        print(f"  {name}: lit fabric RGB {[round(v) for v in b]} -> "
              f"{[round(v) for v in af]}   scale "
              f"{['x%.2f' % k for k in rec['scale']]}"
              + (f"   plate {rec['plate_before']:.0f} -> {rec['plate_after']:.0f}"
                 if rec.get("plate_whitened") else
                 f"   plate {rec['plate_before']:.0f} (left)"))
        if rec.get("mask_grown_pct"):
            print(f"  mask: {rec['mask_note']}")
    else:
        print(f"  {name}: colour left alone - {rec.get('why')}")

    # Measured against the ORIGINAL source like everything else, with the
    # reference beside it so the lay numbers are on the same line.
    try:
        ref = C.reference_path(run)
        m = M.compare(source, out, reference=ref if ref.exists() else None)
        rec["metrics"] = m
        print("  " + M.line(m, out.stem))
    except Exception as e:  # noqa: BLE001 - a measurement is not the delivery
        print(f"  (could not measure {out.name}: {e})")

    (arch / "last_recolour.json").write_text(json.dumps(rec, indent=2,
                                                       default=str) + "\n")
    print(f"  saved {out.name}")
    C.log(run, f"recoloured {name} -> {out.stem}"
               + (" (plate whitened)" if rec.get("plate_whitened") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
