# Gap DAM scraper

## Setup

From the repository root:

```bash
./setup.sh
```

## One command

From the repository root, `./run.sh --style 853417012 ...` does everything
below in order - checks the saved sign-in and prompts for one when it has
lapsed, downloads the style's shots into `inputs/reference_library/853417/`,
and runs the harness against them. The commands that follow are the same steps
run one at a time.

## Sign in

```bash
cd dam_scraper
uv run --locked python dam_auth.py capture
```

Enter your Gap SSO login ID and password at the prompts. The command runs
Chromium headlessly and stores only the authenticated browser state. To inspect
a login failure in a browser window, add `--headed`.

## Download

```bash
uv run --locked python dam_scrape.py 853417012
```

The scraper searches with the first six digits and requires `FINAL` assets. It
takes up to three `P01` images, then falls back to `AV5`. If neither exists, it
takes up to three images from each available Shot Request ID.

The JPGs land in `inputs/reference_library/`. The source ZIP and the manifest
stay in `downloads/<first-six-digits>/`.

Override either location with `--image-root` and `--output-root`, or with the
`DAM_IMAGE_ROOT` and `DAM_OUTPUT_ROOT` environment variables.

If the DAM session expires, run the sign-in command again.

## Docker

From the repository root:

```bash
docker build -t gap-dam-scraper dam_scraper

docker volume create gap-dam-auth

docker run --rm -it --init --ipc=host \
  -v gap-dam-auth:/home/pwuser/.dam-auth \
  --entrypoint python \
  gap-dam-scraper dam_auth.py capture

docker run --rm --init --ipc=host \
  -e DAM_OUTPUT_ROOT=/downloads \
  -v gap-dam-auth:/home/pwuser/.dam-auth:ro \
  -v "$PWD/dam_scraper/downloads:/downloads" \
  -v "$PWD/inputs/reference_library:/images" \
  gap-dam-scraper 853417012
```

The first command captures authentication from an interactive terminal without
a GUI. Later scraper runs reuse the state in the Docker volume.
