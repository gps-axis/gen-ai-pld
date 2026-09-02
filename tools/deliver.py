#!/usr/bin/env python3
"""Move a finished run's deliverables to the output folder, in the names the
callers expect. Run by docker-entrypoint.sh after harness.py; runnable by hand.

    python tools/deliver.py --run runs/<stamp> --out out/ --rc 0 --budget 10

WHAT COMES OUT, and who reads it:

  <out>/best.png, best_2.png ..     the ranked picks (default names), OR
  <out>/generated_1.png ..          the same under --pattern 'generated_{n}.png',
                                    which is what the Kestra flows declare
  <out>/used_prompt.txt             the prompt the rank-1 pick was generated
                                    with; the flows post it to Axis
  <out>/result_top_matches.jpg      the reference near-miss strip under its
  <out>/match_results.json          old names; the flows upload these when a
                                    run is parked for want of a reference
  <out>/result.json                 the receipt; `outcome` is the field to
                                    route on
  <out>/logs/                       the text artefacts and the two images the
                                    model actually worked from
  <run>/output/pickN_cand_XX.png    a copy per pick under the OLD naming, which
                                    the flows' debug bundle parses to learn
                                    which candidate won which slot
  <run>/archive/metrics.json        one row per candidate in the shape the
                                    flows' debug manifest reads (.candidates[]
                                    with .cand and .score), score = lay match
                                    to the reference as a percentage

Every write is independent and guarded: a missing prompt snapshot costs
used_prompt.txt, not the delivery. The Kestra flows were written against the
retired pipeline in old/, and this is where its contract is kept rather than
in the flows, so that Axis and its webhook payloads never had to change.

Exit codes: 0 delivered (or nothing to deliver and the run said so)
            3 the harness exited 0 but nothing reached <out>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

GEN_RE = re.compile(r"^cand_\d+$")
ANY_RE = re.compile(r"^cand_\d+[spc]*$")


def generation_of(name: str) -> str:
    m = re.match(r"(cand_\d+)", name or "")
    return m.group(1) if m else name


def load_json(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def ranked_picks(run: Path) -> list[dict]:
    """[{rank, candidate, file}] from output/picks.json, or best.png alone."""
    outdir = run / "output"
    rec = load_json(outdir / "picks.json", {})
    picks = [p for p in rec.get("picks", []) if (outdir / p.get("file", "")).exists()]
    if picks:
        return sorted(picks, key=lambda p: p["rank"])
    if (outdir / "best.png").exists():
        best = load_json(run / "archive" / "best.json", {}).get("candidate")
        return [{"rank": 1, "candidate": best or "best", "file": "best.png",
                 "chosen_by": "model" if best else "harness"}]
    return []


def copy_picks(run: Path, out: Path, picks: list[dict], pattern: str | None,
               rank1_name: str | None) -> list[str]:
    delivered = []
    for p in picks:
        src = run / "output" / p["file"]
        if pattern:
            name = pattern.replace("{n}", str(p["rank"]))
        elif p["rank"] == 1:
            name = rank1_name or "best.png"
        else:
            name = f"best_{p['rank']}.png"
        shutil.copy2(src, out / name)
        delivered.append(name)
    return delivered


def write_pick_names(run: Path, picks: list[dict]) -> list[str]:
    """output/pick1_cand_10.png ... - the retired grader's naming, which the
    flows' debug bundle parses with `pick(N).*_(cand_NN).png`."""
    written = []
    for p in picks:
        gen = generation_of(p["candidate"])
        if not GEN_RE.match(gen):
            continue
        dst = run / "output" / f"pick{p['rank']}_{gen}.png"
        if not dst.exists():
            shutil.copy2(run / "output" / p["file"], dst)
        written.append(dst.name)
    return written


def write_metrics_json(run: Path, picks: list[dict]) -> Path | None:
    """archive/metrics.json in the retired grader's shape: .candidates[] with
    .cand and .score. score is the lay match to the reference, 0-100, so the
    debug view's number means the thing the pick was actually judged on."""
    arch = run / "archive"
    src = arch / "source_clean.jpg"
    if not src.exists():
        return None
    import metrics as M
    ref = C.reference_path(run)
    ref = ref if ref.exists() else None
    picked = {generation_of(p["candidate"]): p["rank"] for p in picks}
    rows = []
    for p in sorted(arch.glob("cand_*.png")):
        if not GEN_RE.match(p.stem):
            continue
        row = {"cand": p.stem, "picked_as": picked.get(p.stem)}
        try:
            m = M.compare(src, p, reference=ref)
            lay = m.get("lay_iou")
            row.update({
                "score": round(lay * 100, 1) if lay is not None else None,
                "lay_iou": lay, "lay_iou_source": m.get("lay_iou_source"),
                "size_vs_ref": m.get("size_vs_ref"),
                "flat": m.get("wrinkle_ratio"),
                "colour_de": m.get("colour_de_lit"), "hue_de": m.get("hue_de"),
                "same_garment_iou": m.get("silhouette_iou"),
                "unreliable": bool(m.get("cue_note")),
            })
        except Exception as e:  # noqa: BLE001 - one bad image must not stop the rest
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    out = arch / "metrics.json"
    out.write_text(json.dumps({
        "candidates": rows,
        "score": "lay match to the reference silhouette, 0-100 (lay_iou x 100)",
        "reference": str(ref) if ref else None,
        "picks": picks,
    }, indent=2, default=str) + "\n")
    return out


def used_prompt(run: Path, picks: list[dict]) -> str:
    """The prompt the rank-1 pick was generated with, from its snapshot. The
    snapshot opens with an HTML comment carrying the system prompt; that line
    is dropped, since the flows post this text to a person."""
    arch = run / "archive"
    snap = None
    if picks:
        gen = generation_of(picks[0]["candidate"])
        h = (load_json(arch / "lineage.json", {}).get(gen) or {}).get("prompt_hash")
        if h and (arch / f"prompt_{h}.txt").exists():
            snap = arch / f"prompt_{h}.txt"
    if snap is None:
        snaps = sorted(arch.glob("prompt_*.txt"), key=lambda p: p.stat().st_mtime)
        snap = snaps[-1] if snaps else None
    if snap is None:
        try:
            import promptfile as PF
            return PF.render(run)
        except Exception:  # noqa: BLE001
            return ""
    text = snap.read_text()
    if text.startswith("<!--"):
        end = text.find("-->")
        if end >= 0:
            text = text[end + 3:].lstrip("\n")
    return text


def result_json(run: Path, out: Path, session: str, rc: int, budget: int,
                delivered: list[str], picks: list[dict]) -> dict:
    status, summary, best = None, "", None
    t = run / "transcript.jsonl"
    if t.exists():
        for line in t.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") in ("finish", "end"):
                d = rec.get("data", {})
                d = d.get("result", d)
                status = d.get("status", status)
                summary = d.get("summary", summary) or summary
                best = d.get("best") or best
            if rec.get("kind") == "end":
                best = rec.get("data", {}).get("shipped") or best
    attempts = sorted(p.stem for p in (run / "archive").glob("cand_*.png")
                      if GEN_RE.match(p.stem))
    reference = None
    d = load_json(run / "reference_selection.json", None)
    if isinstance(d, dict):
        reference = {
            "match_found": d.get("match_found"),
            "source": d.get("source"),
            "score": d.get("score"),
            "threshold": d.get("threshold"),
            "library_count": d.get("library_count"),
            "disqualified": len(d.get("disqualified") or []),
            "closest": d.get("closest"),
            "model_confidence": d.get("model_confidence"),
            "model_vetoed": d.get("model_vetoed"),
            "differences": d.get("differences"),
            "construction_risk": (d.get("construction_risk") or {}).get("terms"),
            "tone": d.get("tone"),
        }
    outcome = {
        "done": "delivered",
        "budget_exhausted": "delivered_at_budget",
        "gave_up": "nothing_shippable",
        "no_candidates": "no_candidates",
    }.get(status)
    if outcome is None:
        # The run ended before the agent had a status: the two step-0 exits
        # are answers, everything else non-zero is an error.
        outcome = {0: "delivered", 20: "no_reference",
                   21: "unclean_source"}.get(rc, "error")
    rec = {
        "session": session,
        "outcome": outcome,
        "exit_code": rc,
        "status": status,
        "summary": summary,
        "best": picks[0]["candidate"] if picks else best,
        "images_used": len(attempts),
        "budget": budget,
        "attempts": attempts,
        "picks": len(delivered),
        "ranked": [{"rank": p["rank"], "candidate": p["candidate"],
                    "chosen_by": p.get("chosen_by")} for p in picks],
        "images": delivered,
        "reference": reference,
        # Retired with the candidate grader. Held as null rather than dropped so
        # a downstream parser that reads it keeps working; the lay numbers per
        # candidate are in archive/metrics.json.
        "grades": None,
    }
    (out / "result.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session", default="")
    ap.add_argument("--rc", type=int, default=0, help="harness.py's exit code")
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--pattern", default=None,
                    help="name for the ranked picks with {n} for the rank, "
                         "e.g. generated_{n}.png. Default: best.png, best_2.png ..")
    ap.add_argument("--rank1-name", default=None,
                    help="name for rank 1 when no --pattern (default best.png)")
    ap.add_argument("--ship-candidates", action="store_true",
                    help="copy every attempt beside the picks")
    a = ap.parse_args()

    run, out = a.run.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    session = a.session or run.name

    picks = ranked_picks(run)
    delivered = copy_picks(run, out, picks, a.pattern, a.rank1_name)

    if a.ship_candidates:
        for cand in sorted((run / "archive").glob("cand_*.png")):
            if ANY_RE.match(cand.stem):
                shutil.copy2(cand, out / cand.name)
                delivered.append(cand.name)

    # The retired pipeline's names, for the flows that still read them.
    pick_names = write_pick_names(run, picks) if picks else []
    metrics = None
    try:
        metrics = write_metrics_json(run, picks)
    except Exception as e:  # noqa: BLE001 - bookkeeping never blocks a delivery
        print(f"deliver: metrics.json not written ({e})", file=sys.stderr)
    (out / "used_prompt.txt").write_text(used_prompt(run, picks))
    for old, new in (("result_top_matches.jpg", "reference_match.jpg"),
                     ("match_results.json", "reference_selection.json")):
        src = run / new
        if src.exists():
            shutil.copy2(src, out / old)
            if not (run / old).exists():
                shutil.copy2(src, run / old)

    for f in ("steps.log", "LOG.md", "run.log", "transcript.jsonl",
              "reference_selection.json", "reference_match.jpg",
              "archive/lineage.json", "archive/prompt_sections.json",
              "archive/notes.json", "archive/best.json", "archive/seeds.json",
              "archive/metrics.json", "output/picks.json",
              "archive/source_clean.jpg", "archive/reference_greyscale.jpg",
              "archive/reference_original.jpg", "archive/reference.jpg"):
        src = run / f
        if src.exists():
            shutil.copy2(src, out / "logs" / src.name)
    shutil.copy2(out / "used_prompt.txt", out / "logs" / "used_prompt.txt")

    result_json(run, out, session, a.rc, a.budget, delivered, picks)

    print(f"  delivered {len(delivered)} file(s) to {out}"
          + (f": {', '.join(delivered)}" if delivered else ""))
    if pick_names:
        print(f"  old names {', '.join(pick_names)} in {run / 'output'}")
    if metrics:
        print(f"  metrics   {metrics}")
    print(f"  logs      {out / 'logs'}")
    if a.rc == 0 and not delivered:
        print(f"deliver: the harness exited 0 but nothing reached {out}. See "
              f"{out / 'logs' / 'steps.log'} and result.json.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
