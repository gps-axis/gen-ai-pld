# Gap DAM scraper

## Setup

From the repository root:

```bash
./setup.sh
```

## Sign in

```bash
cd dam_scraper
uv run --locked python dam_auth.py capture --channel chrome
```

Complete the login in Chrome. When the Gap DAM page loads, return to the
terminal and press Enter.

## Download

```bash
uv run --locked python dam_scrape.py 853417012
```

The scraper searches with the first six digits. It takes up to three `P01`
images, then falls back to `AV5`. If neither exists, it takes up to three images
from each available Shot Request ID.

Downloads go to `downloads/<first-six-digits>/`.

If the DAM session expires, run the sign-in command again.

## Docker

From the repository root:

```bash
docker build -t gap-dam-scraper dam_scraper

docker run --rm --init --ipc=host \
  -e DAM_AUTH_STATE=/run/secrets/dam-auth.json \
  -e DAM_OUTPUT_ROOT=/downloads \
  -v "$PWD/dam_scraper/secrets/dam-auth.json:/run/secrets/dam-auth.json:ro" \
  -v "$PWD/dam_scraper/downloads:/downloads" \
  gap-dam-scraper 853417012
```

Capture `dam-auth.json` before starting the container. Copy it to the server
through a secure channel.
