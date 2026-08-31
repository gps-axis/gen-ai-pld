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
# TWO inputs are needed now: the garment photo and the reference laydown to aim
# at. The reference used to be found by searching a 45-image library baked into
# the image; that search is gone and the caller supplies it. By default it is
# whatever /in/reference*.{jpg,png} matches, or name it with -e REFERENCE=... .
#
# Anything after the image path is passed straight to harness.py.
#
# Exit codes:
#
#    0   the run delivered
#    1   the run broke, or had a budget and produced nothing
#    2   this script was called wrong (no image, no reference, missing credential)
#    3   the harness reported success but nothing reached /out
#   20   RESERVED AND UNREACHABLE. It used to mean "no library reference is close
#        enough to this garment, upload a hero". There is no library and no
#        search, so nothing returns it. Kept documented rather than deleted so a
#        caller that still branches on it keeps parsing.
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
    # maxdepth 1 keeps inputs/others/ out of it, and anything named reference* is
    # the OTHER input rather than a candidate for this one.
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
REFERENCE="${REFERENCE:-}"
if [ -z "$REFERENCE" ]; then
    mapfile -t REFS < <(find "$IN_DIR" -maxdepth 1 -type f -iname 'reference*' \
        ! -name '.*' 2>/dev/null | sort)
    [ ${#REFS[@]} -gt 0 ] && REFERENCE="${REFS[0]}"
fi
[ -n "$REFERENCE" ] || die "no reference laydown.
  Mount one at $IN_DIR/reference_greyscale.jpg, or name it with -e REFERENCE=...
  It is the flat the garment is being laid out to match. There is no library to
  search any more - the caller chooses it."
[ -f "$REFERENCE" ] || die "no such reference: $REFERENCE"

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
# The extension is preserved because PIL opens by content but writes by suffix,
# and the harness desaturates this file on its way into the run folder.
REF_EXT="${REFERENCE##*.}"
REF_STAGED="/app/inputs/reference_input.${REF_EXT:-jpg}"
cp "$REFERENCE" "$REF_STAGED" || die "could not stage $REFERENCE"

echo "  input     $INPUT"
echo "  reference $REFERENCE"
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
    --reference "$REF_STAGED" \
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
    # Retired with the reference search and the grader. Held as null rather than
    # dropped so a downstream parser that reads them keeps working.
    "reference": None,
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
         archive/lineage.json archive/prompt_sections.json \
         archive/notes.json archive/best.json archive/seeds.json; do
    [ -e "$RUN_DIR/$f" ] && cp -p "$RUN_DIR/$f" "$OUT_DIR/logs/"
done
# The two images the run was actually about, so a reviewer never has to guess
# what the model was looking at.
for f in source_clean.jpg reference.jpg; do
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

# Unreachable, and kept so that if it ever DOES fire the script says something
# true rather than falling through to a bare exit code nobody can place.
if [ "$RC" -eq "$EXIT_NO_REFERENCE" ]; then
    echo
    echo "  outcome   exit 20, which this pipeline no longer produces - there is"
    echo "            no reference search left to fail. Treat it as a bug."
    exit "$NO_REFERENCE_EXIT"
fi

exit "$RC"
