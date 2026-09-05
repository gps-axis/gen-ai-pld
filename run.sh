#!/usr/bin/env bash
# One command for the whole job:
#
#     ./run.sh --style 853417012 --source inputs/IMG_4281.HEIC --max-images 10 --yolo
#
# In order: make sure the Gap DAM sign-in is still good (and sign in from the
# terminal when it is not), download that style's laydown shots into the
# reference library, decode the phone's HEIC, then run the harness against
# those shots. Before this file did the last two, the operator did the first
# two by hand in another folder, and the only thing linking the four steps was
# remembering to do them in that order.
#
# --style is the only flag consumed here. Everything else goes to harness.py
# untouched. Without --style nothing touches the DAM and the command is exactly
# what it always was.
#
# The harness runs on an interpreter that has numpy/PIL/scipy/requests/
# fal_client and pillow_heif - the last one is what lets --source take a .HEIC.
# Order matters: the project-local .venv comes first, because neither
# /usr/bin/python3 nor the Homebrew python3 has any of the libraries, and a bare
# `python3` produces an ImportError several seconds into a run rather than a
# clear message here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { echo "run.sh: $*" >&2; exit 1; }

for CAND in "$HERE/.venv/bin/python" "$HERE/../env3.10/bin/python3" "$(command -v python3 || true)"; do
  if [ -n "$CAND" ] && [ -x "$CAND" ] && "$CAND" -c 'import numpy, PIL, scipy, requests, fal_client, pillow_heif' 2>/dev/null; then
    PY="$CAND"
    break
  fi
done

if [ -z "${PY:-}" ]; then
  echo "No interpreter with the required libraries was found." >&2
  echo "Build the project venv, then retry:" >&2
  echo "    ./setup.sh" >&2
  exit 1
fi

# --- pull --style out of the arguments -------------------------------------
STYLE=""
HAS_REFERENCE=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --style)   [ $# -ge 2 ] || die "--style needs a value"; STYLE="$2"; shift 2 ;;
    --style=*) STYLE="${1#--style=}"; shift ;;
    --reference|--reference=*|--reference-library|--reference-library=*)
               HAS_REFERENCE=1; ARGS+=("$1"); shift ;;
    *)         ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# One run = one folder. The pipeline scripts key their run folder off this, so
# the agent cannot start a second folder to get a second helping of the image
# budget. Set it yourself to resume an existing folder.
export LAYDOWN_SESSION="${LAYDOWN_SESSION:-$(date +%Y%m%d_%H%M%S)}"
export LAYDOWN_MAX_IMAGES="${LAYDOWN_MAX_IMAGES:-10}"
# The ceiling, not necessarily this run's budget - --max-images lowers it, and
# the harness prints what it actually settled on a few lines later.
echo "  session   $LAYDOWN_SESSION  (ceiling $LAYDOWN_MAX_IMAGES images)"

# --- the DAM: sign-in, then that style's shots --------------------------------
#
# Same rules as dam_scraper/dam_scrape.py: 6 digits is the style, 9 is style
# plus colour, and the DAM is searched on the first six either way. The shots
# land flat in inputs/reference_library/ next to every other style's shots -
# no per-style folder - and the harness picks its reference from the whole
# library. Which files belong to which style is recorded in the manifest under
# dam_scraper/downloads/<style>/, not in the folder layout.
if [ -n "$STYLE" ]; then
  [ "$HAS_REFERENCE" = 0 ] || die "--style chooses the reference from the DAM; drop --reference / --reference-library, or drop --style."
  [[ "$STYLE" =~ ^[0-9]{6}([0-9]{3})?$ ]] || die "--style must be 6 or 9 digits, got '$STYLE'"
  command -v uv >/dev/null 2>&1 || die "uv is required for the DAM steps - run ./setup.sh first."
  DAM="$HERE/dam_scraper"
  LIB="$HERE/inputs/reference_library"
  MANIFEST="$DAM/downloads/${STYLE:0:6}/manifest.json"
  # VIRTUAL_ENV is unset for these so uv uses the scraper's own environment
  # (playwright lives there, not in the harness venv) instead of warning about
  # the one that happens to be activated in the shell.
  dam() { env -u VIRTUAL_ENV uv run --locked --project "$DAM" python "$DAM/$1" "${@:2}"; }

  echo "  style     $STYLE  -> ${LIB#"$HERE/"}/  (flat, alongside every other style)"

  # check exits 0 when the saved session still reaches the DAM, 2 when there is
  # no saved session, 3 when there is one and the DAM sent it back to login.
  # The last two have the same fix; anything else is Playwright or Chromium
  # broken, and a password prompt is not the answer to that.
  set +e; dam dam_auth.py check; RC=$?; set -e
  case $RC in
    0) ;;
    2|3) [ -t 0 ] || die "the DAM sign-in has expired and there is no terminal to sign in from. Run: cd dam_scraper && uv run --locked python dam_auth.py capture"
         echo "  signing in to the Gap DAM"
         dam dam_auth.py capture || die "DAM sign-in failed (exit $?)" ;;
    *) die "dam_auth.py check broke (exit $RC) - see the message above" ;;
  esac

  # Already-downloaded styles come back from the zip without touching the
  # network; dam_scrape.py keeps a manifest per style and knows the difference.
  # --output-root is spelled out because the scraper's default is relative to
  # the working directory, and this command is run from anywhere.
  dam dam_scrape.py "$STYLE" --image-root "$LIB" --output-root "$DAM/downloads" \
    || die "DAM download failed (exit $?)"
  # The library is shared by every style, so "holds a JPG" proves nothing about
  # this one. Check the files the manifest says this style produced, by name,
  # directly under the library root.
  "$PY" - "$MANIFEST" "$LIB" <<'PYCHECK' || die "dam_scrape.py reported success but ${LIB#"$HERE/"}/ is missing this style's JPGs (see above)"
import json, sys
from pathlib import Path
manifest, lib = Path(sys.argv[1]), Path(sys.argv[2])
names = [f["filename"] for f in json.loads(manifest.read_text()).get("files", [])]
missing = [n for n in names if not (lib / n).is_file()]
if not names:
    print(f"  {manifest} lists no files", file=sys.stderr); sys.exit(1)
if missing:
    print("  missing: " + ", ".join(missing), file=sys.stderr); sys.exit(1)
print(f"  library   {len(names)} JPG(s) for this style, flat in {lib.name}/")
PYCHECK
  set -- --reference-library "$LIB" "$@"
fi

# The skill is the operating manual and there is only one, so it does not need
# naming on every invocation. An explicit --skill or --skill-file still wins.
SKILL_ARGS=()
case " $* " in
  *" --skill "*|*" --skill-file "*) ;;
  *) [ -f "$HERE/task/SKILL.md" ] && SKILL_ARGS=(--skill-file "$HERE/task/SKILL.md") ;;
esac

exec "$PY" "$HERE/harness.py" "${SKILL_ARGS[@]}" "$@"
