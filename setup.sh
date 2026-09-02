#!/usr/bin/env bash
# Prepare both Python environments and the DAM browser runtime. Authentication
# stays a separate, interactive step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAM_PROJECT="$ROOT/dam_scraper"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 1
fi

echo "Syncing the root project..."
uv sync --locked --project "$ROOT"

echo "Syncing the DAM scraper..."
uv sync --locked --project "$DAM_PROJECT"

echo "Installing Chromium for Playwright..."
uv run --locked --project "$DAM_PROJECT" playwright install chromium

echo "Verifying the DAM scraper..."
uv run --locked --project "$DAM_PROJECT" python "$DAM_PROJECT/verify.py"

echo
echo "Setup complete. DAM authentication has not been captured."
echo "When ready, run:"
echo "  cd \"$DAM_PROJECT\""
echo "  uv run --locked python dam_auth.py capture"
