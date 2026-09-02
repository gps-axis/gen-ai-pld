#!/usr/bin/env bash
# Two images in, one re-laid flat out.
#
# This exists so the host never has to know the harness's internal layout. The
# harness works from inputs/ next to its own source and writes to a timestamped
# runs/<session>/ folder, so the caller would otherwise have to bind-mount over
# the source tree and go hunting. Here the caller mounts photos at /in and a
# folder at /out.
#
#   docker run --rm -v "$PWD/inputs:/in:ro" -v "$PWD/out:/out" \
#     -v pld-cache:/app/.cache -e FAL_KEY -e QWEN_API_KEY \
#     pld-harness /in/off_set_image.jpg
#
# The garment photo is required. The reference laydown it is being laid out to
# match comes from one of two places, and an explicit one always wins:
#
#   * mount a library at /in/reference_library (or -e REFERENCE_LIBRARY=...) and
#     the harness chooses the reference itself, for free
#   * mount a single reference matching /in/reference* (or -e REFERENCE=...) and
#     no search happens
#
# Anything after the image path is passed straight to harness.py.
#
# Exit codes:
#
#    0   the run delivered
#    1   the run broke, or had a budget and produced nothing
#    2   this script was called wrong (no image, no reference AND no library,
#        missing credential)
#    3   the harness reported success but nothing reached /out
#   20   the library was searched and holds nothing close enough to this garment.
#        A real answer, not a crash: nothing was generated and nothing billed.
#        Remap it with NO_REFERENCE_EXIT.
#   21   the segmenter returned an image with most of the garment missing
set -uo pipefail

PY=/app/.venv/bin/python
IN_DIR="${IN_DIR:-/in}"
OUT_DIR="${OUT_DIR:-/out}"

# Mirrored from harness.py so the two cannot drift.
EXIT_NO_REFERENCE=20
EXIT_UNCLEAN_SOURCE=21
NO_REFERENCE_EXIT="${NO_REFERENCE_EXIT:-$EXIT_NO_REFERENCE}"

die() { echo "entrypoint: $*" >&2; exit 2; }

# --- which image are we laying out? ----------------------------------------
#
# An explicit path wins. With none, /in must hold exactly one candidate image:
# guessing between several would mean silently laying out the wrong garment, and
# the run costs real money at fal.ai.
INPUT=""
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    INPUT="$1"
    shift
    [ -f "$INPUT" ] || die "no such image: $INPUT"
else
    [ -d "$IN_DIR" ] || die "no image given and $IN_DIR is not mounted."
    # maxdepth 1 keeps both inputs/others/ and the reference library out of it -
    # a library holds hundreds of garment photos, and every one of them would
    # otherwise look like a candidate for the thing being laid out. Anything
    # named reference* is the OTHER input rather than a candidate for this one.
    mapfile -t FOUND < <(find "$IN_DIR" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
           -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \) \
        ! -name '.*' ! -iname 'reference*' | sort)
    case ${#FOUND[@]} in
        0) die "no garment image found in $IN_DIR (looked for jpg/png/tif/webp,
  ignoring anything named reference*)." ;;
        1) INPUT="${FOUND[0]}" ;;
        *) printf 'entrypoint: %d images in %s - name the one you want:\n' \
               "${#FOUND[@]}" "$IN_DIR" >&2
           printf '  %s\n' "${FOUND[@]}" >&2
           exit 2 ;;
    esac
fi

# --- and what are we laying it out LIKE? ------------------------------------
#
# An explicit reference wins outright; otherwise a mounted library is searched.
# The find below deliberately excludes the library folder itself, so that
# `-v .../reference_library:/in/reference_library` is not picked up as a single
# reference file by the `reference*` glob.
REFERENCE="${REFERENCE:-}"
if [ -z "$REFERENCE" ]; then
    mapfile -t REFS < <(find "$IN_DIR" -maxdepth 1 -type f -iname 'reference*' \
        ! -name '.*' 2>/dev/null | sort)
    [ ${#REFS[@]} -gt 0 ] && REFERENCE="${REFS[0]}"
fi

REFERENCE_LIBRARY="${REFERENCE_LIBRARY:-$IN_DIR/reference_library}"
LIB_COUNT=0
if [ -z "$REFERENCE" ] && [ -d "$REFERENCE_LIBRARY" ]; then
    LIB_COUNT=$(find "$REFERENCE_LIBRARY" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
        ! -name '.*' 2>/dev/null | wc -l | tr -d ' ')
fi

if [ -n "$REFERENCE" ]; then
    [ -f "$REFERENCE" ] || die "no such reference: $REFERENCE"
elif [ "$LIB_COUNT" -eq 0 ]; then
    die "no reference laydown and no library to find one in.
  Either mount a library at $IN_DIR/reference_library (or point
  -e REFERENCE_LIBRARY at one), or mount a single reference at
  $IN_DIR/reference_greyscale.jpg and name it with -e REFERENCE=... .
  The reference is the flat the garment is being laid out to match."
fi

# --- how many images may this run buy? --------------------------------------
#
# 0 is a legitimate and useful setting: the loop runs, the model writes a prompt,
# generate refuses, and nothing is billed. It is how you smoke-test a deployment.
export LAYDOWN_SESSION="${LAYDOWN_SESSION:-$(date +%Y%m%d_%H%M%S)}"
export LAYDOWN_MAX_IMAGES="${LAYDOWN_MAX_IMAGES:-10}"

# --- credentials ------------------------------------------------------------
#
# Checked here rather than discovered 20 turns in, when the model finally calls
# generate and the run has already spent its context. A zero-budget run never
# reaches fal, so it is not asked for the key - that is the point of running one.
if [ "$LAYDOWN_MAX_IMAGES" -gt 0 ]; then
    [ -n "${FAL_KEY:-}" ] || die "FAL_KEY is not set - pass it with -e FAL_KEY.
  (LAYDOWN_MAX_IMAGES=0 runs the loop without it and bills nothing.)"
fi
[ -n "${QWEN_API_KEY:-}" ] || die "QWEN_API_KEY is not set - pass it with -e QWEN_API_KEY.
  It is deliberately not baked into the image."

RUN_DIR="/app/runs/$LAYDOWN_SESSION"
mkdir -p /app/inputs "$RUN_DIR" "$OUT_DIR" || die "cannot write to $OUT_DIR"

# Copied rather than symlinked: /in is usually mounted read-only and the harness
# writes derived images beside these.
cp "$INPUT" /app/inputs/off_set_image.jpg || die "could not stage $INPUT"

# Either --reference <file> or --reference-library <dir>, never both: the
# harness triggers its search on the ABSENCE of --reference, so passing an empty
# one would silently disable the library.
REF_ARGS=()
if [ -n "$REFERENCE" ]; then
    # The extension is preserved because PIL opens by content but writes by
    # suffix, and the harness desaturates this file into the run folder.
    REF_EXT="${REFERENCE##*.}"
    REF_STAGED="/app/inputs/reference_input.${REF_EXT:-jpg}"
    cp "$REFERENCE" "$REF_STAGED" || die "could not stage $REFERENCE"
    REF_ARGS=(--reference "$REF_STAGED")
else
    # Read in place. A library is hundreds of files and copying it would double
    # the container's disk for no gain; /in being read-only is fine, because
    # nothing writes to the library - the descriptions land in /app/.cache.
    REF_ARGS=(--reference-library "$REFERENCE_LIBRARY")
fi

echo "  input     $INPUT"
if [ -n "$REFERENCE" ]; then
    echo "  reference $REFERENCE"
else
    echo "  reference searching $REFERENCE_LIBRARY ($LIB_COUNT images)"
fi
echo "  session   $LAYDOWN_SESSION  (max $LAYDOWN_MAX_IMAGES images)"
echo "  text      ${QWEN_BASE_URL:-http://10.11.245.41:8091 (default)}"
echo "  segmenter ${SEGMENT_URL:-http://10.11.245.145:4000/sam3-segment (default)}"
echo

# --yolo is required, not a preference: Approver.ok() refuses every mutating tool
# when stdin is not a TTY, so without it the model is denied on its first real
# step and burns the run arguing with itself.
"$PY" /app/harness.py \
    --skill-file /app/task/SKILL.md \
    --source /app/inputs/off_set_image.jpg \
    "${REF_ARGS[@]}" \
    --yolo "$@"
RC=$?

# --- deliver ----------------------------------------------------------------
#
# Run unconditionally: a run that failed halfway still leaves a candidate worth
# looking at, and always leaves the logs that explain what went wrong.
DELIVER=()
if [ -e "$RUN_DIR/output/best.png" ]; then
    cp -p "$RUN_DIR/output/best.png" "$OUT_DIR/${OUTPUT_NAME:-best.png}"
    DELIVER+=("${OUTPUT_NAME:-best.png}")
fi

# SHIP_CANDIDATES=1 delivers every attempt beside the winner, so a reviewer can
# see what was rejected. Named by their run-folder stems, which are stable and
# already carry the ordering the model generated them in.
if [ -n "${SHIP_CANDIDATES:-}" ] && [ -d "$RUN_DIR/archive" ]; then
    for cand in "$RUN_DIR"/archive/cand_*.png; do
        [ -e "$cand" ] || continue
        cp -p "$cand" "$OUT_DIR/$(basename "$cand")"
        DELIVER+=("$(basename "$cand")")
    done
fi

# Settled before result.json is written, so the receipt and the exit code cannot
# disagree. A harness that reported success while nothing reached /out is an
# entrypoint-level delivery failure and has its own code.
SHORT=""
if [ "$RC" -eq 0 ] && [ ${#DELIVER[@]} -eq 0 ]; then
    SHORT=1
fi

# `status`, `attempts` and `images_used` come from the transcript rather than
# being re-derived here: the harness already resolved them, including the
# coercion that turns finish("done") with nothing generated into no_candidates.
"$PY" - "$RUN_DIR" "$OUT_DIR" "$LAYDOWN_SESSION" "$RC" "$LAYDOWN_MAX_IMAGES" \
    > "$OUT_DIR/result.json" <<'PY'
import json, pathlib, re, sys

run, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
session, rc, budget = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

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
                  if re.fullmatch(r"cand_\d+", p.stem))

# The reference receipt, when the library was searched. Deliberately a summary
# and not the whole record - the full thing, including every disqualified
# candidate and the per-field breakdown, stays at reference_selection.json.
# `null` when the reference was supplied by hand: there was no choice to report.
reference = None
sel = run / "reference_selection.json"
if sel.exists():
    try:
        d = json.loads(sel.read_text())
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
        }
    except (ValueError, OSError):
        reference = None

# `outcome` is the field a caller routes on; `exit_code` is for whoever is
# debugging rather than routing. They are not the same question: "the model
# judged nothing here shippable" is a real answer that happens to end the run.
outcome = {
    "done": "delivered",
    "budget_exhausted": "delivered_at_budget",
    "gave_up": "nothing_shippable",
    "no_candidates": "no_candidates",
}.get(status, "error" if rc else "delivered")

rec = {
    "session": session,
    "outcome": outcome,
    "exit_code": rc,
    "status": status,
    "summary": summary,
    "best": best,
    "images_used": len(attempts),
    "budget": budget,
    "attempts": attempts,
    # `picks` is kept, and is 1 or 0: this pipeline delivers one image now, not a
    # ranked four. A caller that counted picks still counts something true.
    "picks": 1 if (out / "best.png").exists() else 0,
    "images": sorted(p.name for p in out.glob("*.png")),
    # null when the reference was supplied rather than found - see above.
    "reference": reference,
    # Retired with the candidate grader. Held as null rather than dropped so a
    # downstream parser that reads it keeps working.
    "grades": None,
}
print(json.dumps(rec, indent=2, default=str))
PY

# The text artefacts are the only way to explain a run that shipped nothing, so
# they come out even when runs/ is not mounted. Every copy is guarded: three of
# the files this used to take unconditionally belonged to the reference search
# and no longer exist, and `cp` of a missing file is a non-zero exit inside a
# script whose whole job is to report the run's own status.
mkdir -p "$OUT_DIR/logs"
for f in steps.log LOG.md run.log transcript.jsonl \
         reference_selection.json reference_match.jpg \
         archive/lineage.json archive/prompt_sections.json \
         archive/notes.json archive/best.json archive/seeds.json; do
    [ -e "$RUN_DIR/$f" ] && cp -p "$RUN_DIR/$f" "$OUT_DIR/logs/"
done
# The two images the run was actually about, so a reviewer never has to guess
# what the model was looking at.
# reference.jpg is the pre-rename name; both are listed so a run folder from
# either era delivers its reference rather than silently omitting it.
for f in source_clean.jpg reference_greyscale.jpg reference_original.jpg \
         reference.jpg; do
    [ -e "$RUN_DIR/archive/$f" ] && cp -p "$RUN_DIR/archive/$f" "$OUT_DIR/logs/"
done

echo
echo "  delivered ${#DELIVER[@]} file(s) to $OUT_DIR"
echo "  logs      $OUT_DIR/logs"

if [ -n "$SHORT" ]; then
    echo "entrypoint: the harness exited 0 but nothing reached $OUT_DIR." >&2
    echo "  See $OUT_DIR/logs/steps.log and result.json for what the run did." >&2
    exit 3
fi

if [ "$RC" -eq "$EXIT_UNCLEAN_SOURCE" ]; then
    echo
    echo "  outcome   the segmenter returned an image missing most of the"
    echo "            garment. Nothing was generated from it, on purpose."
    echo "  look at   $OUT_DIR/logs/run.log"
fi

if [ "$RC" -eq "$EXIT_NO_REFERENCE" ]; then
    echo
    echo "  outcome   the library holds nothing close enough to this garment."
    echo "            Nothing was generated and nothing was billed."
    echo "  next      add a suitable laydown to $REFERENCE_LIBRARY, or supply"
    echo "            one directly with -e REFERENCE=/in/my_reference.jpg"
    echo "  detail    $OUT_DIR/result.json (.reference.closest) and"
    echo "            $OUT_DIR/logs/reference_match.jpg"
    exit "$NO_REFERENCE_EXIT"
fi

exit "$RC"
