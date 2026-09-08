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
# Anything after the image path is passed straight to harness.py, except four
# things the Kestra flows still send from the retired pipeline, which are
# translated here so that Axis and its webhook payloads never had to change:
#
#   --reference-category CAT    search only <library>/CAT (falls back to the
#                               whole library when that folder is missing)
#   --no-reference-select       use the reference the flow dropped at
#                               /app/inputs/reference_greyscale.jpg; no search
#   --task "...grade_flats..."  a task naming a retired script is dropped, with
#                               a warning; the harness has its own default
#   OUTPUT_PATTERN=generated_{n}.png
#                               name the ranked picks that way in $OUT_DIR
#                               instead of best.png, best_2.png ..
#
# And two that hand the run to run.sh, the operator's own command line:
#
#   --style 671304002           pull this style's laydown shots from the Gap
#                               DAM into the library first, then search it
#   --item-details "GAP MENS MENS KNITS L/S KNITS"
#                               the text the DAM is searched with when the
#                               style has no shots, and the harness's last
#                               resort when none of the style's shots match
#
# The DAM sign-in comes from DAM_LOGIN_ID/DAM_PASSWORD (or the files their
# _FILE forms name); the saved session and the scraper's manifests live at
# DAM_AUTH_STATE and DAM_OUTPUT_ROOT, both under /app and worth mounting so
# they outlive the container. An explicit reference still wins over both
# flags: the DAM is not searched when the operator has supplied the answer.
#
# tools/deliver.py writes everything the flows read after the run: the picks,
# used_prompt.txt, result_top_matches.jpg, match_results.json, result.json, the
# pickN_cand_XX.png names in the run's output/ and archive/metrics.json.
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
           -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.webp' \
           -o -iname '*.heic' -o -iname '*.heif' \) \
        ! -name '.*' ! -iname 'reference*' | sort)
    case ${#FOUND[@]} in
        0) die "no garment image found in $IN_DIR (looked for jpg/png/tif/webp/heic,
  ignoring anything named reference*)." ;;
        1) INPUT="${FOUND[0]}" ;;
        *) printf 'entrypoint: %d images in %s - name the one you want:\n' \
               "${#FOUND[@]}" "$IN_DIR" >&2
           printf '  %s\n' "${FOUND[@]}" >&2
           exit 2 ;;
    esac
fi

# --- the retired pipeline's flags, translated -------------------------------
CATEGORY=""
NO_SELECT=""
TASK=""
HAS_TASK=""
STYLE=""
ITEM_DETAILS=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --reference-category)   CATEGORY="${2:-}"; shift 2 ;;
        --reference-category=*) CATEGORY="${1#*=}"; shift ;;
        --no-reference-select)  NO_SELECT=1; shift ;;
        --task)                 TASK="${2:-}"; HAS_TASK=1; shift 2 ;;
        --task=*)               TASK="${1#*=}"; HAS_TASK=1; shift ;;
        --style)                STYLE="${2:-}"; shift 2 ;;
        --style=*)              STYLE="${1#*=}"; shift ;;
        --item-details)         ITEM_DETAILS="${2:-}"; shift 2 ;;
        --item-details=*)       ITEM_DETAILS="${1#*=}"; shift ;;
        *)                      ARGS+=("$1"); shift ;;
    esac
done
if [ -n "$HAS_TASK" ]; then
    if [ -z "$TASK" ]; then
        echo "entrypoint: --task is empty; using the skill's own goal" >&2
    elif printf '%s' "$TASK" | grep -qiE 'grade_flats|stage_batch|--ship|match_reference'; then
        echo "entrypoint: --task names a retired tool and is dropped:" >&2
        echo "  $TASK" >&2
        echo "  The harness delivers the best four on its own now." >&2
    else
        ARGS+=(--task "$TASK")
    fi
fi
set -- "${ARGS[@]+"${ARGS[@]}"}"

# --- and what are we laying it out LIKE? ------------------------------------
#
# An explicit reference wins outright; otherwise a mounted library is searched.
# The find below deliberately excludes the library folder itself, so that
# `-v .../reference_library:/in/reference_library` is not picked up as a single
# reference file by the `reference*` glob.
REFERENCE="${REFERENCE:-}"
if [ -n "$NO_SELECT" ] && [ -z "$REFERENCE" ]; then
    # The commands flow copies the operator's approved reference here before
    # calling, then asks for no search.
    REFERENCE="/app/inputs/reference_greyscale.jpg"
    [ -f "$REFERENCE" ] || die "--no-reference-select given but $REFERENCE is missing.
  Copy the reference there first, or pass -e REFERENCE=<file>."
fi
if [ -z "$REFERENCE" ]; then
    mapfile -t REFS < <(find "$IN_DIR" -maxdepth 1 -type f -iname 'reference*' \
        ! -name '.*' 2>/dev/null | sort)
    [ ${#REFS[@]} -gt 0 ] && REFERENCE="${REFS[0]}"
fi

# --- the Gap DAM: this style's own laydown shots, pulled in first ----------
#
# --style / --item-details hand the run to run.sh, which signs in to the DAM,
# downloads the style's laydown shots flat into the library, and runs the
# harness against the whole library - the command an operator runs on a
# laptop, with the library at DAM_IMAGE_ROOT instead of inputs/. The category
# folder is not narrowed on this path: the style's own shots are the most
# specific references there are, they land at the library root, and the
# search recurses into every category folder as well. An explicit reference
# wins over both flags, as it does over the library.
DAM_PULL=""
if [ -n "$STYLE" ] || [ -n "$ITEM_DETAILS" ]; then
    if [ -n "$REFERENCE" ]; then
        echo "  dam       --style/--item-details ignored: an explicit reference was supplied"
        STYLE=""; ITEM_DETAILS=""
    else
        DAM_PULL=1
    fi
fi

# The library: -e REFERENCE_LIBRARY, else /in/reference_library, else the path
# the Kestra flows mount the shared library at.
if [ -z "${REFERENCE_LIBRARY:-}" ]; then
    if [ -d "$IN_DIR/reference_library" ]; then
        REFERENCE_LIBRARY="$IN_DIR/reference_library"
    else
        REFERENCE_LIBRARY="/app/library_reference"
    fi
fi
if [ -n "$CATEGORY" ] && [ -n "$DAM_PULL" ]; then
    echo "  category  $CATEGORY not narrowed: the DAM pull searches the whole library"
elif [ -n "$CATEGORY" ] && [ -z "$REFERENCE" ]; then
    # Axis already knows the category; searching only its folder is what the
    # retired matcher did with the same flag. A folder that does not exist or
    # holds nothing falls back to the whole library rather than to exit 20.
    SUB="$REFERENCE_LIBRARY/$CATEGORY"
    SUB_COUNT=$(find "$SUB" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
        ! -name '.*' 2>/dev/null | wc -l | tr -d ' ')
    if [ "$SUB_COUNT" -gt 0 ]; then
        REFERENCE_LIBRARY="$SUB"
        echo "  category  $CATEGORY -> searching $SUB ($SUB_COUNT images)"
    else
        echo "  category  $CATEGORY has no folder or no images under $REFERENCE_LIBRARY; searching the whole library"
    fi
fi
LIB_COUNT=0
if [ -z "$REFERENCE" ] && [ -d "$REFERENCE_LIBRARY" ]; then
    LIB_COUNT=$(find "$REFERENCE_LIBRARY" -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
        ! -name '.*' 2>/dev/null | wc -l | tr -d ' ')
fi

if [ -n "$REFERENCE" ]; then
    [ -f "$REFERENCE" ] || die "no such reference: $REFERENCE"
elif [ -n "$DAM_PULL" ]; then
    # The pull creates the library if it has to; an empty one is not an error
    # yet. run.sh checks the login before anything else and says what it
    # needs when none is stored.
    mkdir -p "$REFERENCE_LIBRARY" || die "cannot create the library at $REFERENCE_LIBRARY"
    [ -w "$REFERENCE_LIBRARY" ] || die "the library at $REFERENCE_LIBRARY is read-only, and the DAM pull writes into it.
  Mount it writable (-v .../reference_library:/app/library_reference), or point
  -e REFERENCE_LIBRARY at a folder that is."
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
elif [ -n "$DAM_PULL" ]; then
    echo "  reference the Gap DAM${STYLE:+, style $STYLE}${ITEM_DETAILS:+, text \"$ITEM_DETAILS\"}; then $REFERENCE_LIBRARY ($LIB_COUNT images before the pull)"
    echo "  dam       session ${DAM_AUTH_STATE:-dam_scraper/secrets/dam-auth.json}, manifests ${DAM_OUTPUT_ROOT:-dam_scraper/downloads}"
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
if [ -n "$DAM_PULL" ]; then
    # run.sh adds --reference-library itself (and refuses one passed in), and
    # exits 1 before the harness when the DAM cannot be reached or signed in
    # to. A DAM that merely holds nothing for the style is a warning there,
    # and the harness goes on with the library as it stands.
    DAM_ARGS=()
    [ -z "$STYLE" ] || DAM_ARGS+=(--style "$STYLE")
    [ -z "$ITEM_DETAILS" ] || DAM_ARGS+=(--item-details "$ITEM_DETAILS")
    DAM_IMAGE_ROOT="$REFERENCE_LIBRARY" /app/run.sh "${DAM_ARGS[@]}" \
        --skill-file /app/task/SKILL.md \
        --source /app/inputs/off_set_image.jpg \
        --yolo "$@"
else
    "$PY" /app/harness.py \
        --skill-file /app/task/SKILL.md \
        --source /app/inputs/off_set_image.jpg \
        "${REF_ARGS[@]}" \
        --yolo "$@"
fi
RC=$?

# --- deliver ----------------------------------------------------------------
#
# Run unconditionally: a run that failed halfway still leaves a candidate worth
# looking at, and always leaves the logs that explain what went wrong. Every
# name the flows read - generated_N.png, used_prompt.txt, result_top_matches.jpg,
# match_results.json, result.json, pickN_cand_XX.png, archive/metrics.json -
# is written by tools/deliver.py, so the contract lives in one testable place.
DELIVER_ARGS=(--run "$RUN_DIR" --out "$OUT_DIR" --session "$LAYDOWN_SESSION"
              --rc "$RC" --budget "$LAYDOWN_MAX_IMAGES")
[ -n "${OUTPUT_PATTERN:-}" ] && DELIVER_ARGS+=(--pattern "$OUTPUT_PATTERN")
[ -n "${OUTPUT_NAME:-}" ]    && DELIVER_ARGS+=(--rank1-name "$OUTPUT_NAME")
[ -n "${SHIP_CANDIDATES:-}" ] && DELIVER_ARGS+=(--ship-candidates)
echo
"$PY" /app/tools/deliver.py "${DELIVER_ARGS[@]}"
DRC=$?

if [ "$DRC" -eq 3 ]; then
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
