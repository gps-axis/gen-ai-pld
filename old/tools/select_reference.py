#!/usr/bin/env python3
"""Step 0 - choose the lay reference, deterministically, before the agent runs.

    python tools/select_reference.py --run runs/20260817_140159

Picking the reference used to be a human step: look at the off-set photo, find
the closest garment in library_reference/, desaturate it, drop it in inputs/.
This does exactly that, and does it the same way every time.

It is two pieces:

  match_reference.py   scores every library image against the query and returns
                       ONE winner or an honest "no match". It is the judgement.
  this file            installs that winner - greyscale, full resolution - at
                       the single path the pipeline reads, records where it came
                       from, and gets out of the way. It is the plumbing.

Two things it insists on, both because of how the rest of the pipeline behaves:

  * GREYSCALE. prepare.py branches on the reference's mode: `L` makes the brief
    tell the model to ignore the reference's tone, anything else warns that
    colour may bleed from it. The reference is a shape and construction
    reference only, so it is desaturated here rather than being left as a
    second, competing colour source. --colour opts out.
  * ONE reference at a time. The agent reads its inputs off a fingerprinted
    inventory, and two files called reference-something are two candidates. Any
    other reference*.jpg in inputs/ is moved aside into inputs/others/ (moved,
    not deleted) unless --no-stash.

Exit codes:  0 installed   2 no match, nothing installed   1 broke

2 is a business outcome, not a fault: the library holds nothing close enough to
this garment and a human has to upload a hero. It is written down as well as
returned - reference_selection.json is produced for BOTH outcomes, carrying
`match_found`, so a caller that routes on the answer has one file that always
exists and always answers the same question. A receipt that appeared only on
success would be indistinguishable from a run that died before step 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

import common as C

Image.MAX_IMAGE_PIXELS = None

MATCHER = Path(__file__).with_name("match_reference.py")

# Words that name how a garment is BUILT rather than how it is laid out.
#
# The matcher's `differences` line is the only sentence in the pipeline that
# says what the reference has and the product does not, and it is written
# before a penny is spent. On runs/20260819_205617 it read "the candidate
# features a slightly more defined V-neckline and visible seam piping along the
# edges" - and four of the ten candidates came back with a neckline seam and
# topstitching down the straps, flagged by stage 3, at 15c each. The reference
# is supposed to contribute pose and nothing else, so ANY construction word in
# that line is a risk worth naming: it is not a claim that the run will fail,
# it is the one early warning available.
CONSTRUCTION_TERMS = (
    "seam", "seams", "seaming", "piping", "stitch", "stitching", "topstitch",
    "topstitching", "topstitched", "panel", "panels", "panelling", "paneling",
    "neckline", "binding", "bound edge", "trim", "dart", "darts", "gusset",
    "pocket", "pockets", "zip", "zipper", "closure", "logo", "label", "mesh",
    "cutout", "cut-out", "overlay", "elastic", "hem", "cuff", "waistband",
)


def construction_terms(text: str | None) -> list[str]:
    """Which construction words a differences line uses, in the order given."""
    if not text:
        return []
    low = str(text).lower()
    found = []
    for t in CONSTRUCTION_TERMS:
        # Whole words only: 'seam' must not fire on 'seamless', which is a
        # fabric finish and the opposite of a construction difference.
        if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low) and t not in found:
            found.append(t)
    return found

# The one path the pipeline reads. prepare.py takes --reference explicitly and
# the SKILL tells the agent to pass it, so this name only has to be stable, not
# guessed - but it stays the name the workspace has always used, so an old
# command line still works.
CANON = "reference_greyscale.jpg"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def stash_other_references(inputs: Path, keep: str) -> list[str]:
    """Move any other reference*.* out of inputs/ and into inputs/others/.

    Moved, never deleted: one of these is usually the hand-picked reference from
    a previous garment, and it is the operator's file, not ours.
    """
    others = inputs / "others"
    moved = []
    for p in sorted(inputs.glob("reference*")):
        if not p.is_file() or p.name == keep:
            continue
        others.mkdir(parents=True, exist_ok=True)
        dst = others / p.name
        n = 1
        while dst.exists():
            dst = others / f"{p.stem}_prev{n}{p.suffix}"
            n += 1
        shutil.move(str(p), str(dst))
        moved.append(f"{p.name} -> others/{dst.name}")
    return moved


def install(src: Path, dst: Path, greyscale: bool,
            silhouette: bool = False) -> tuple[bool, str]:
    """Write the winner to dst. Returns (changed, description).

    Skips the write when the bytes would be identical, so a re-run does not
    churn the mtime - the harness fingerprints inputs by md5 and an mtime that
    moves for no reason reads as a changed input.

    `silhouette` installs a line drawing of the reference's outline instead of
    the photograph. Same pose, same arrangement, no construction to copy - see
    common.outline_map.
    """
    tmp_outline = None
    if silhouette:
        tmp_outline = dst.with_name(f".{dst.stem}.outline.jpg")
        C.outline_map(src, tmp_outline)
        src = tmp_outline
    with Image.open(src) as im:
        out = im.convert("L") if greyscale else im.convert("RGB")
    # Hidden, so a temp file left behind by a crash is neither stashed as a
    # stray reference nor listed in the workspace inventory.
    tmp = dst.with_name(f".{dst.stem}.tmp.jpg")
    out.save(tmp, quality=95, subsampling=0)
    changed = not (dst.exists() and md5(dst) == md5(tmp))
    if changed:
        tmp.replace(dst)
    else:
        tmp.unlink()
    if tmp_outline and tmp_outline.exists():
        tmp_outline.unlink()
    with Image.open(dst) as check:
        desc = f"{check.width}x{check.height} mode={check.mode}"
    return changed, desc


def check_runner_up(query: Path, runner: dict, library: Path,
                    a) -> tuple[Path | None, str]:
    """Ask the model the same question about the runner-up. One vision call.

    Returns (path to install, why) - path None means keep the winner. It is
    kept unless the runner-up is BOTH still a genuine style match and free of
    construction words, because swapping a named risk for an unmeasured one is
    not an improvement. On the batch this was written for the runner-up scored
    97.1 against the winner's 99.2, so the demotion is cheap when it is
    available and worth refusing when it is not.
    """
    name = runner.get("_file")
    if not name:
        return None, "no runner-up in the ranking"
    hits = sorted(library.rglob(name))
    if len(hits) != 1:
        return None, f"cannot resolve {name!r} under {library.name}"
    alt = hits[0]
    if runner.get("score", 0) < a.threshold:
        return None, (f"{name} scores {runner.get('score', 0):.1f}, under the "
                      f"{a.threshold:.0f} threshold")
    try:
        import match_reference as M
        client = M.Client(M._default_base_url(), "", 180)
        client.resolve_model()
        v = M.compare_multi(client, M.ensure_small(query, 1024),
                            [("A", M.ensure_small(alt, 1024))])
    except Exception as e:  # noqa: BLE001 - a failed check keeps the winner
        return None, f"check failed ({type(e).__name__}: {e}); keeping the winner"

    if str(v.get("pick", "")).strip().lower() in ("none", ""):
        return None, f"{name} is not the same style ('none'); keeping the winner"
    alt_terms = construction_terms(v.get("differences"))
    if alt_terms:
        return None, (f"{name} names construction too "
                      f"({', '.join(alt_terms)}); keeping the winner")
    return alt, (f"{name} differs in no construction term "
                 f"({v.get('differences') or 'no differences given'})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, default=C.INPUTS / "off_set_image.jpg",
                    help="the off-set photo the reference has to match. The "
                         "harness passes <run>/archive/offset_upload.jpg, the "
                         "pre-cleaned image, so the query and the library are "
                         "both garments on a white plate.")
    ap.add_argument("--query-cleaned", action="store_true",
                    help="declare that --query has been pre-cleaned. Recorded "
                         "in reference_selection.json; without it the run is "
                         "marked as matched against a raw input and says so.")
    ap.add_argument("--library", type=Path, default=C.ROOT / "library_reference")
    ap.add_argument("--inputs", type=Path, default=C.INPUTS,
                    help="folder the chosen reference is installed into")
    ap.add_argument("--name", default=CANON,
                    help=f"filename to install as (default {CANON})")
    ap.add_argument("--run", type=Path, default=None,
                    help="run folder for the provenance record and steps.log "
                         "(default: this session's folder)")
    ap.add_argument("--category", default=None,
                    help="force a library subfolder instead of letting the "
                         "query's own garment_type pick one")
    # 95 rather than the matcher's 90; see the note in harness.py's
    # --reference-threshold for what that buys and what it gives up. harness.py
    # passes this explicitly on every run, so this default only applies to a
    # direct call of this script - and it is kept in step with the harness so
    # the two cannot quietly disagree about what a reference has to score.
    ap.add_argument("--threshold", type=float, default=95.0,
                    help="score a library image must reach to be installed "
                         "(default 95)")
    ap.add_argument("--colour", "--color", dest="colour", action="store_true",
                    help="install the reference in colour; the default "
                         "desaturates it so it cannot act as a colour target")
    ap.add_argument("--silhouette", action="store_true",
                    help="install a line drawing of the winner's OUTLINE "
                         "instead of the photograph. It carries the pose - "
                         "straps crossed, band flat, legs closed - and no "
                         "construction at all, so there is literally no seam "
                         "or neckline for the generator to copy across. The "
                         "strongest fix for construction bleed, and the one "
                         "that changes what the generator sees, so it is opt-in "
                         "until a paid batch has been run with it.")
    ap.add_argument("--demote-on-bleed", action="store_true",
                    help="when the winner's differences line names construction, "
                         "ask the model the same question about the runner-up "
                         "and install that instead if it comes back clean. One "
                         "extra vision call, no billed images. Keeps the winner "
                         "and says why when the runner-up is no better, which is "
                         "the common case.")
    ap.add_argument("--no-stash", dest="stash", action="store_false",
                    help="leave any other reference* files in inputs/ alone")
    ap.add_argument("--dry-run", action="store_true",
                    help="match and report, install nothing")
    ap.add_argument("--matcher-arg", action="append", default=[], metavar="ARG",
                    help="pass an extra argument straight to match_reference.py "
                         "(repeatable, e.g. --matcher-arg --color-weight "
                         "--matcher-arg 0)")
    a = ap.parse_args()

    query = a.query.resolve()
    library = a.library.resolve()
    if not query.exists():
        print(f"query not found: {query}", file=sys.stderr)
        return 1
    if not library.is_dir():
        print(f"library not found: {library}", file=sys.stderr)
        return 1

    run = (a.run or C.session_run_dir()).resolve()
    run.mkdir(parents=True, exist_ok=True)

    # --- the judgement ---------------------------------------------------
    cmd = [sys.executable, str(MATCHER),
           "--query", str(query),
           "--library", str(library),
           "--out-dir", str(run),
           "--threshold", str(a.threshold)]
    if a.category:
        cmd += ["--category", a.category]
    cmd += a.matcher_arg

    # Flushed, because the child writes straight to the same terminal and an
    # unflushed header lands after all of the matcher's output.
    print(f"reference selection: {query.name} vs {library.name}/"
          f"{'  (pre-cleaned query)' if a.query_cleaned else ''}", flush=True)
    if not a.query_cleaned:
        # Every library image is a garment on a white plate. Scoring a raw phone
        # photo against them charges the query for its room and its hang tag,
        # and the score that comes out is the number the whole run is anchored
        # to. Degraded, not dead - but it has to be visible here and in the
        # receipt, or a raw-match run is indistinguishable from a clean one.
        print("WARNING: matched against raw input - not pre-cleaned. The tag "
              "and the room are being scored as part of the garment.")
    rc = subprocess.run(cmd).returncode
    results = run / "match_results.json"

    if rc == 1 or not results.exists():
        print(f"\nmatcher failed (exit {rc}); no reference installed.", file=sys.stderr)
        return 1

    res = json.loads(results.read_text())

    if rc == 2 or not res.get("match_found"):
        best = (res.get("ranked") or [{}])[0]
        print("\nNO REFERENCE SELECTED - nothing in the library matched this garment.")
        print(f"  closest    {best.get('_file', '?')} at "
              f"{best.get('score', 0):.1f} (needed {a.threshold:.0f})")
        print(f"  detail     {results}")
        print("  nothing was installed; inputs/ is unchanged.")
        # The same receipt as a successful selection, with match_found false.
        # This is the file a caller branches on - Kestra routes a miss to a
        # human upload request rather than failing the flow - so it has to be
        # written on the path that produces no reference, which is exactly the
        # path that used to write nothing at all.
        record = {
            "selected_at": datetime.now().isoformat(timespec="seconds"),
            "match_found": False,
            "query": str(query),
            "query_md5": md5(query),
            "query_cleaned": bool(a.query_cleaned),
            "query_attrs": res.get("query_attrs"),
            "library_root": res.get("library_root"),
            "library_used": res.get("library_used"),
            "library_count": res.get("library_count"),
            "source": None,
            "installed": None,
            "score": None,
            "threshold": res.get("threshold"),
            "model": res.get("model"),
            "model_confidence": res.get("model_confidence"),
            "model_vetoed": res.get("model_vetoed"),
            "n_qualifying": res.get("n_qualifying"),
            # What the library came closest to holding. Whoever is being asked
            # to upload a hero wants to see this and the contact sheet, not a
            # bare "no match" - the near miss is often the argument for
            # widening the category rather than shooting a new garment.
            "closest": {"file": best.get("_file"),
                        "score": best.get("score")},
            "reason": (res.get("verdict") or {}).get("reason"),
            "match_results": str(results),
            "contact_sheet": str(run / "result_top_matches.jpg"),
        }
        (run / "reference_selection.json").write_text(
            json.dumps(record, indent=2, default=str))
        C.log(run, f"reference NOT selected (best "
                   f"{best.get('score', 0):.1f} < {a.threshold:.0f})")
        return 2

    src = Path(res["match_path"]) if res.get("match_path") else None
    if not src or not src.exists():
        # Older matcher records carried only the basename. Fall back to finding
        # it, but say so - a silent guess about which category folder a file
        # came from is exactly the class of error this pipeline is built around.
        hits = sorted(library.rglob(res["match"]))
        if len(hits) != 1:
            print(f"\ncannot resolve {res['match']!r} to one file under "
                  f"{library} ({len(hits)} candidates)", file=sys.stderr)
            return 1
        src = hits[0]
        print(f"  (resolved {res['match']} by search: {src})")

    # --- the plumbing ----------------------------------------------------
    dst = (a.inputs / a.name).resolve()
    verdict = res.get("verdict") or {}
    ranked = res.get("ranked") or []
    runner = next((r for r in ranked if r.get("_file") != res["match"]), {})

    # --- construction bleed ----------------------------------------------
    # The reference's job is the LAY. Anything it also carries - a neckline
    # shape, piping, a seam - is something the generator can copy into a
    # product that does not have it, and that copy is indistinguishable from a
    # real feature by the time anyone looks at the output.
    bleed = {"flagged": False, "terms": [], "line": verdict.get("differences"),
             "action": "none", "detail": ""}
    terms = construction_terms(verdict.get("differences"))
    if terms:
        bleed.update({"flagged": True, "terms": terms})
        print(f"\nCONSTRUCTION BLEED RISK - the pick differs from the product in "
              f"{len(terms)} construction term(s): {', '.join(terms)}")
        print(f"  differences: {verdict.get('differences')}")
        print("  The reference is a LAY reference. Anything it shows that the "
              "product does not have can be copied into the generation, and "
              "on a real run four of ten candidates were flagged for exactly "
              "the seams this line named.")

    if terms and a.demote_on_bleed and runner.get("_file"):
        alt, why = check_runner_up(query, runner, library, a)
        bleed["detail"] = why
        print(f"  runner-up check: {why}")
        if alt is not None:
            src, bleed["action"] = alt, "demoted to the runner-up"
            print(f"  DEMOTED: installing {alt.name} instead "
                  f"({runner.get('score', 0):.1f} vs {res.get('match_score')})")
        else:
            bleed["action"] = "kept the winner; the runner-up was no cleaner"
    elif terms and a.demote_on_bleed:
        bleed["action"] = "kept the winner; no runner-up to fall back on"
    elif terms:
        bleed["action"] = ("silhouette installed, so nothing can be copied"
                           if a.silhouette else "warned only")

    if a.dry_run:
        print(f"\nDRY RUN - would install {src} -> {dst}"
              f"{'' if a.colour else ' (as greyscale)'}"
              f"{' (as an outline map)' if a.silhouette else ''}")
        return 0

    a.inputs.mkdir(parents=True, exist_ok=True)
    moved = stash_other_references(a.inputs, a.name) if a.stash else []
    changed, desc = install(src, dst, greyscale=not a.colour,
                            silhouette=a.silhouette)

    record = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        # Stated on both paths so a caller can branch on one key without having
        # to infer the outcome from whether `installed` came back null.
        "match_found": True,
        # Which image was actually scored, by path and by content. A run matched
        # against the cleaned upload and one matched against the raw phone photo
        # produce the same-shaped record, and this is what tells them apart
        # afterwards.
        "query": str(query),
        "query_md5": md5(query),
        "query_cleaned": bool(a.query_cleaned),
        "query_attrs": res.get("query_attrs"),
        "library_root": res.get("library_root"),
        "library_used": res.get("library_used"),
        "library_count": res.get("library_count"),
        "source": str(src),
        "source_md5": md5(src),
        "installed": str(dst),
        "installed_md5": md5(dst),
        "installed_desc": desc,
        "greyscale": not a.colour,
        "rewritten": changed,
        "stashed": moved,
        "score": res.get("match_score"),
        "threshold": res.get("threshold"),
        "model": res.get("model"),
        "model_confidence": res.get("model_confidence"),
        "model_vetoed": res.get("model_vetoed"),
        "n_qualifying": res.get("n_qualifying"),
        "runner_up": {"file": runner.get("_file"), "score": runner.get("score")},
        "reason": verdict.get("reason"),
        "differences": verdict.get("differences"),
        # What the reference can leak into the generation, decided before any
        # image is paid for. generate.py reads this and hardens the prompt with
        # the specific words; prepare.py puts them in the brief the agent
        # writes from.
        "construction_risk": bleed,
        "silhouette": bool(a.silhouette),
        "match_results": str(results),
        "contact_sheet": str(run / "result_top_matches.jpg"),
    }
    prov = run / "reference_selection.json"
    prov.write_text(json.dumps(record, indent=2, default=str))

    print("\nREFERENCE SELECTED")
    print(f"  matched vs   {query.name}"
          + ("  (pre-cleaned)" if a.query_cleaned else
             "  (RAW INPUT - tag and background included in the score)"))
    print(f"  source       {src.parent.name}/{src.name}")
    print(f"  score        {res.get('match_score')}/100"
          + (f"   model confidence {res['model_confidence']}"
             if res.get("model_confidence") is not None else ""))
    print(f"  installed    {dst}"
          + ("   <- OUTLINE MAP, not the photograph" if a.silhouette else ""))
    print(f"               {desc}  md5:{record['installed_md5'][:8]}"
          f"{'' if changed else '  (already identical, not rewritten)'}")
    if runner:
        print(f"  runner-up    {runner.get('_file')} "
              f"({runner.get('score', 0):.1f})")
    if record["differences"]:
        # The one line of the model's prose worth reading: what still differs
        # between the query and the pick. The checklist has no field for most of
        # it, so this is the only place a real difference gets named.
        print(f"  differences  {record['differences']}")
    if bleed["flagged"]:
        print(f"  bleed risk   {', '.join(bleed['terms'])}  ->  {bleed['action']}")
        if not a.silhouette and bleed["action"] in ("warned only",
                                                    "kept the winner; the runner-up was no cleaner",
                                                    "kept the winner; no runner-up to fall back on"):
            print("               generate.py will name these in the prompt and "
                  "tell the model not to copy them. Stronger, if this keeps "
                  "happening: re-run this with --silhouette, which installs an "
                  "outline and leaves nothing to copy.")
    for m in moved:
        print(f"  stashed      {m}")
    print(f"  provenance   {prov}")

    C.log(run, f"reference {src.name[:28]} ({res.get('match_score')}/100)"
               + ("" if a.query_cleaned else " vs RAW query")
               + (f", bleed risk: {','.join(bleed['terms'][:3])}"
                  if bleed["flagged"] else "")
               + (" [outline]" if a.silhouette else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
