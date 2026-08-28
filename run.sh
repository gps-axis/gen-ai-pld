#!/usr/bin/env bash
# Run the harness with an interpreter that has numpy/PIL/scipy/requests/fal_client.
#
# Order matters. The project-local .venv comes first: since this project moved
# out of the Axis tree, neither /usr/bin/python3 (3.9.6) nor the Homebrew
# python3 (3.14.3) has any of the libraries, and the old ../env3.10 is no longer
# next door. Falling through to a bare `python3` produces an ImportError several
# seconds into a run rather than a clear message here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for CAND in "$HERE/.venv/bin/python" "$HERE/../env3.10/bin/python3" "$(command -v python3 || true)"; do
  if [ -n "$CAND" ] && [ -x "$CAND" ] && "$CAND" -c 'import numpy, PIL, scipy, requests, fal_client' 2>/dev/null; then
    PY="$CAND"
    break
  fi
done

if [ -z "${PY:-}" ]; then
  echo "No interpreter with the required libraries was found." >&2
  echo "Build the project venv, then retry:" >&2
  echo "    /opt/homebrew/opt/python@3.10/bin/python3.10 -m venv \"$HERE/.venv\"" >&2
  echo "    \"$HERE/.venv/bin/python\" -m pip install -r \"$HERE/requirements.txt\"" >&2
  exit 1
fi

# One run = one folder. The pipeline scripts key their run folder off this, so
# the agent cannot start a second folder to get a second helping of the image
# budget. Set it yourself to resume an existing folder.
export LAYDOWN_SESSION="${LAYDOWN_SESSION:-$(date +%Y%m%d_%H%M%S)}"
export LAYDOWN_MAX_IMAGES="${LAYDOWN_MAX_IMAGES:-10}"
# The ceiling, not necessarily this run's budget - --max-images lowers it, and
# the harness prints what it actually settled on a few lines later.
echo "  session   $LAYDOWN_SESSION  (ceiling $LAYDOWN_MAX_IMAGES images)"

# The skill is the operating manual and there is only one, so it does not need
# naming on every invocation. An explicit --skill or --skill-file still wins.
SKILL_ARGS=()
case " $* " in
  *" --skill "*|*" --skill-file "*) ;;
  *) [ -f "$HERE/task/SKILL.md" ] && SKILL_ARGS=(--skill-file "$HERE/task/SKILL.md") ;;
esac

exec "$PY" "$HERE/harness.py" "${SKILL_ARGS[@]}" "$@"
