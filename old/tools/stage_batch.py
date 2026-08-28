#!/usr/bin/env python3
"""Buy images a few at a time, and stop as soon as enough of them are good.

    python tools/stage_batch.py --run runs/<stamp> --target 4

Generation is the only billed step in this pipeline and the grade is free, so
the order that costs least is: generate a few, grade them, and buy more only if
too few cleared. Nothing here is new work - it runs generate.py and
grade_flats.py exactly as you would - it just refuses to spend the next 15c
before looking at the last ones.

What it replaces: a run generated ten images in one wave, then discovered that
the batch was decided by its fourth. Six of those ten were bought after the
answer was already on disk. On that batch this script would have stopped at four
or six, because five of the first six cleared.

Rules it follows, so a round is never spent on a hunch:

  * The first round is --first images (default 4, the delivery size). If enough
    of those pass, that is the whole run.
  * A top-up buys the SHORTFALL and no more. Buying "a few extra to be safe" is
    the behaviour this file exists to stop; --yield-scale opts into sizing the
    round by the pass rate observed so far when a batch is genuinely weak.
  * A round that adds images and no passes ends the run. Independent draws from
    the same prompt fail the same way, so the next round buys more of the same -
    the fix is the prompt, not the budget.
  * generate.py's own ceiling still applies and is never raised from here.

Every child's output is printed as it happens: this script decides how many to
buy, and the tools it calls decide everything else.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import common as C

HERE = Path(__file__).resolve().parent
PRICE = {"1K": 0.15, "2K": 0.15, "4K": C.PRICE_4K}


def run_child(cmd: list[str], what: str) -> int:
    print(f"\n{'=' * 74}\n{what}\n{'=' * 74}", flush=True)
    return subprocess.run(cmd, cwd=HERE).returncode


def counts(run: Path) -> tuple[int, int, list[str]]:
    """(images on disk, candidates that PASS, their names).

    PASS is read off the grade's own record rather than recomputed here, so
    this script cannot disagree with grade_flats.py about who cleared.
    """
    arch = run / "archive"
    have = len(list(arch.glob("cand_*.png")))
    f = arch / "grade_results.json"
    if not f.exists():
        return have, 0, []
    try:
        rows = json.loads(f.read_text()).get("candidates") or []
    except (json.JSONDecodeError, OSError):
        return have, 0, []
    names = [r["name"] for r in rows if r.get("status") == "PASS"]
    return have, len(names), names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--target", type=int, default=4,
                    help="how many candidates have to clear the grade before "
                         "the run stops buying (default 4, the delivery size)")
    ap.add_argument("--first", type=int, default=4,
                    help="images in the first round (default 4). Anything "
                         "smaller cannot fill a four-image delivery in one "
                         "round even if every draw is perfect.")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="rounds of generate-then-grade (default 3)")
    ap.add_argument("--yield-scale", action="store_true",
                    help="size a top-up by the pass rate seen so far instead of "
                         "by the bare shortfall. Buys more per round on a weak "
                         "batch, which is faster and costs more; off by default "
                         "because overshooting the target is the exact waste "
                         "this script exists to prevent.")
    ap.add_argument("--resolution", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--reference", type=Path, default=None,
                    help="passed straight to generate.py")
    ap.add_argument("--expected-changes", default="", metavar="TEXT",
                    help="passed straight to grade_flats.py - what the clean "
                         "step legitimately removed, so its absence is not "
                         "graded as a defect")
    ap.add_argument("--min-grade", type=float, default=None,
                    help="passed straight to grade_flats.py")
    ap.add_argument("--ship", action="store_true",
                    help="finish by shipping --target picks with "
                         "grade_flats.py --ship-faithful")
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    arch = a.run / "archive"
    if not (arch / "prompt.txt").exists():
        print(f"No {arch / 'prompt.txt'}. Write the prompt first - "
              f"prepare.py leaves the brief in archive/prompt_brief.md.",
              file=sys.stderr)
        return 1

    def grade_cmd(extra: list[str]) -> list[str]:
        cmd = [sys.executable, str(HERE / "grade_flats.py"), "--run", str(a.run)]
        if a.expected_changes:
            cmd += ["--expected-changes", a.expected_changes]
        if a.min_grade is not None:
            cmd += ["--min-grade", str(a.min_grade)]
        return cmd + extra

    have0, passed, _ = counts(a.run)
    spent_rounds, price = 0, PRICE[a.resolution]
    if have0:
        print(f"{have0} image(s) already in {arch.name}; grading those before "
              f"buying anything.")
        run_child(grade_cmd([]), "grade what is already here")
        have0, passed, names = counts(a.run)
        print(f"\n{passed} of {have0} already clear: {', '.join(names) or 'none'}")

    stop = ""
    for rnd in range(1, a.max_rounds + 1):
        have, passed, names = counts(a.run)
        if passed >= a.target:
            stop = f"{passed} clear, target {a.target} met"
            break

        short = a.target - passed
        want = a.first if (rnd == 1 and not have) else short
        if a.yield_scale and have:
            rate = max(passed / have, 0.15)
            want = max(short, min(int(round(short / rate)), short * 3))
            print(f"\n--yield-scale: {passed}/{have} cleared so far "
                  f"({rate:.0%}); buying {want} to add {short}")

        print(f"\n{'#' * 74}\nround {rnd}/{a.max_rounds}: {passed} of "
              f"{a.target} cleared, {have} image(s) bought so far. "
              f"Generating {want} (~{want * price * 100:.0f}c).\n{'#' * 74}")

        cmd = [sys.executable, str(HERE / "generate.py"), "--run", str(a.run),
               "--num", str(want), "--resolution", a.resolution,
               "--concurrency", str(a.concurrency)]
        if a.reference:
            cmd += ["--reference", str(a.reference)]
        rc = run_child(cmd, f"round {rnd}: generate {want} at {a.resolution}")
        after, _, _ = counts(a.run)
        made = after - have
        if rc != 0 and not made:
            stop = ("generate.py refused or failed and added nothing - read its "
                    "output above; nothing here overrides it")
            break
        if not made:
            stop = "the run is at its image ceiling; no more can be bought"
            break
        spent_rounds += 1

        run_child(grade_cmd([]), f"round {rnd}: grade all {after}")
        _, now, names = counts(a.run)
        gained = now - passed
        print(f"\nround {rnd}: {made} image(s) bought, {gained} more cleared "
              f"({now}/{a.target}); clear now: {', '.join(names) or 'none'}")
        if now >= a.target:
            stop = f"{now} clear, target {a.target} met"
            break
        if gained <= 0:
            stop = (f"round {rnd} bought {made} image(s) and none of them "
                    f"cleared. Independent draws from one prompt fail the same "
                    f"way, so the next round buys more of the same - the prompt "
                    f"is what needs changing, not the budget")
            break
    else:
        stop = f"{a.max_rounds} round(s) used"

    have, passed, names = counts(a.run)
    print(f"\n{'=' * 74}")
    print(f"STOPPED: {stop}")
    print(f"  bought  {have} image(s) over {spent_rounds} round(s)  "
          f"~{have * price * 100:.0f}c at {a.resolution}")
    print(f"  clear   {passed}/{a.target}  {', '.join(names) or '(none)'}")
    if passed < a.target:
        print(f"  Short of the target. Do NOT generate more by hand - "
              f"grade_flats.py --run {a.run} --ship-faithful {a.target} ships "
              f"the most faithful of what exists and prints what each one "
              f"carries.")
    C.log(a.run, f"staged batch: {have} image(s), {spent_rounds} round(s), "
                 f"{passed}/{a.target} clear ({stop[:40]})")

    if a.ship:
        return run_child(grade_cmd(["--ship-faithful", str(a.target)]),
                         f"ship {a.target}")
    return 0 if passed >= a.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
