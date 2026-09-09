# Gap DAM scraper

## Setup

From the repository root:

```bash
./setup.sh
```

## One command

From the repository root, `./run.sh --style 853417012 ...` does everything
below in order - checks the saved sign-in and signs in again when it has
lapsed, downloads the style's shots flat into `inputs/reference_library/` (no
per-style folder), and runs the harness against the whole library. The commands
that follow are the same steps run one at a time.

## The sign-in

The scraper needs a Gap SSO login. Give it one in any of these ways and it
signs in by itself whenever the saved session is missing or has lapsed -
nothing to type, in a container or on a laptop:

- **`DAM_LOGIN_ID` and `DAM_PASSWORD`** in the environment. In a container,
  feed them from the orchestrator's secret store (a Kestra secret, or
  `docker run -e`), never from a file in the repository.
- **`DAM_LOGIN_ID_FILE` and `DAM_PASSWORD_FILE`**, each naming a file that
  holds the value - the Docker and Compose secrets convention, where the
  secret is mounted read-only under `/run/secrets/` and only its path is in
  the environment. Prefer this over the plain variables when the environment
  is visible to more than this one process (`docker inspect` shows it). Set
  the value or the file for each setting, not both.
- **The macOS Keychain**, for local runs on a Mac: `uv run --locked python
  dam_auth.py store` asks once and keeps the login as item `gap-dam-sso`.
  `dam_auth.py forget` removes it. The environment wins over the Keychain.

The saved session lives at `DAM_AUTH_STATE` (default
`dam_scraper/secrets/dam-auth.json`; a volume in the container) and is reused
until the DAM rejects it, so a sign-in happens only when it has to. The scraper
never asks at the terminal; with no login stored it stops and says so.

Do not put the login in the repository's `.env`: the harness hands that file
to the agent's shell, and `run.sh` scrubs `DAM_LOGIN_ID` and `DAM_PASSWORD`
from the harness's environment for the same reason.

## Sign in by hand

```bash
cd dam_scraper
uv run --locked python dam_auth.py capture
```

Uses the stored login above when there is one, else asks at the terminal - and
says which. It runs Chromium headlessly and stores only the authenticated
browser state. To inspect a login failure in a browser window, add `--headed`.

## Download

```bash
uv run --locked python dam_scrape.py 853417012
```

The scraper searches with the first six digits and requires `FINAL` assets. It
takes up to ten `P01` images, then falls back to `AV5`. If neither exists, it
takes up to ten images from each available Shot Request ID. The cap is
`MAX_PER_CODE` in `dam_scrape.py`. A style downloaded under a smaller cap is
fetched again on its next run, unless it had already taken every shot the DAM
had for that code.

When the style number has nothing in the DAM, free text can stand in for it:

```bash
uv run --locked python dam_scrape.py --item-details "vintage soft hoodie"
uv run --locked python dam_scrape.py 440760022 --item-details "vintage soft hoodie"
```

`--item-details` searches the DAM with the text and takes the first 50 laydown
results (`ITEM_DETAILS_LIMIT`, one DAM results page) in the DAM's own order,
whatever their Shot Request ID. On its own it is the whole job; next to a style
number it runs only when the style has nothing to download: no laydown assets,
or some but none of them tagged FINAL. Any other failure of the style search
still stops the run. Its ZIP and manifest live under
`downloads/item-details/<text>/`.

The scraper's last line of output is `manifest <path>`, naming whichever
manifest the run produced; `run.sh` reads it from there.

The JPGs land directly in `inputs/reference_library/`, flat: no per-style
folder, and any folder inside the DAM's ZIP is dropped rather than recreated.
The source ZIP and the manifest stay in `downloads/<first-six-digits>/`; the
manifest is what records which files belong to which style.

The DAM sends two or more selected assets as a ZIP and a single one as the
bare JPG, so a style with exactly one laydown shot comes back as that JPG. The
scraper wraps it into a one-file ZIP under the name the DAM gave it, so
`downloads/` and the manifest look the same either way; the manifest marks
such an archive with `wrapped_bare_file`.

Override either location with `--image-root` and `--output-root`, or with the
`DAM_IMAGE_ROOT` and `DAM_OUTPUT_ROOT` environment variables.

If the DAM session expires, the scraper signs in again with the stored login;
without one, run `dam_auth.py capture` by hand.

## Docker

From the repository root:

```bash
docker build -t gap-dam-scraper dam_scraper

docker volume create gap-dam-auth

docker run --rm --init --ipc=host \
  -e DAM_LOGIN_ID -e DAM_PASSWORD \
  -v gap-dam-auth:/home/pwuser/.dam-auth \
  -v "$PWD/dam_scraper/downloads:/downloads" \
  -v "$PWD/inputs/reference_library:/images" \
  gap-dam-scraper 853417012
```

No separate sign-in step: with the login in the environment (or in files named
by `DAM_LOGIN_ID_FILE`/`DAM_PASSWORD_FILE`, mounted from the orchestrator's
secrets), the scraper signs in on its first run and whenever the DAM rejects
the saved session. The volume keeps the session between runs so that happens
only when it has to, and it must be writable for the same reason. The
interactive form still works for a one-off:

```bash
docker run --rm -it --init --ipc=host \
  -v gap-dam-auth:/home/pwuser/.dam-auth \
  --entrypoint python \
  gap-dam-scraper dam_auth.py capture
```
