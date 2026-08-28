# Running the harness in a container

Two images in, one re-laid flat out.

    docker build -t pld-harness .

    docker run --rm \
      -v "$PWD/inputs:/in:ro" \
      -v "$PWD/out:/out" \
      -e FAL_KEY="$(grep -o 'FAL_KEY.*' .env | cut -d= -f2 | tr -d ' \"')" \
      -e QWEN_API_KEY="$(cat .qwen_key)" \
      -e LAYDOWN_MAX_IMAGES=10 \
      pld-harness /in/off_set_image.jpg

`out/best.png` is the delivery. `out/logs/` gets `steps.log`, `LOG.md`,
`run.log`, `transcript.jsonl`, the run's `lineage.json`, `prompt_sections.json`,
`notes.json`, and the two images the model was actually working from
(`source_clean.jpg` and `reference.jpg`).

## Two inputs, and the reference is yours now

The reference laydown used to be found by searching a 45-image library baked
into the image. That search is gone, the library is not shipped, and the caller
supplies the reference.

By default the entrypoint takes whatever matches `/in/reference*`; name it
explicitly with `-e REFERENCE=/in/my_reference.jpg`. The garment photo is either
the path you pass as the first argument, or - if you pass none - the single
image in `/in` that is *not* named `reference*`. It refuses to guess between
several: choosing the wrong one costs a full run at fal.ai.

Anything after the image path goes straight to `harness.py`:

    docker run ... pld-harness /in/photo.jpg --max-iters 60 --max-images 4

## The container is a client, not a system

Nothing is bundled but the code. Three services stay outside it, and each has to
be reachable from inside the container:

| what | env var | default |
|---|---|---|
| text/vision model (the agent) | `QWEN_BASE_URL` | `http://10.11.245.41:8091` |
| segmentation service | `SEGMENT_URL` | `http://10.11.245.41:8780/segment` |
| fal.ai (the billed generation) | `FAL_KEY` | none - required unless the budget is 0 |

`QWEN_API_KEY` is required. On the host it comes from `.qwen_key`; that file and
`.env` are in `.dockerignore` on purpose, because anyone who can pull an image
can read its layers. Pass them with `-e`.

`REFMATCH_BASE_URL` is no longer read by anything - it belonged to the reference
matcher. Passing it is harmless and ignored.

The LAN endpoints resolve from inside the container over Docker's NAT with no
extra flags.

## Volumes

`/in` can be read-only. `/app/inputs` inside the container cannot, which is why
the entrypoint copies your photos in rather than mounting over that folder.

Add `-v "$PWD/runs:/app/runs"` to keep full run folders, including every
generated candidate in `archive/`. Without it you get `best.png` and the text
artefacts, and the attempts that were not chosen go with the container. Add
`-e SHIP_CANDIDATES=1` to have every attempt copied to `/out` alongside the
winner instead.

`-v pld-cache:/app/.cache` is no longer worth much - the attribute cache it
existed for belonged to the reference matcher. Harmless to keep.

**On colima:** a bind mount only works if the host path is one colima shares
with its VM - `$HOME` by default. A folder under `/tmp` silently appears empty
inside the container.

## Image budget

`LAYDOWN_MAX_IMAGES` defaults to 10, matching `run.sh`. It is a ceiling for the
whole run, not a target - the model stops when the result is good enough.

**`LAYDOWN_MAX_IMAGES=0` is a supported and useful setting.** The loop runs, the
model looks at both images and writes a prompt, `generate` refuses and says the
budget is zero, and nothing is billed. `FAL_KEY` is not required. It is the way
to smoke-test a deployment - endpoints, mounts, credentials, the model itself -
for free, and it exits 0.

## Exit codes

| code | meaning |
|---|---|
| 0 | delivered |
| 1 | the run broke, or had a budget and produced nothing |
| 2 | misconfigured: missing credential, no image, no reference, several images |
| 3 | the harness reported success but nothing reached `/out` |
| 20 | **reserved and unreachable** - see below |
| 21 | the segmenter returned an image with most of the garment missing |

### 20 can no longer occur

It used to mean "nothing in the library is close enough to this garment, a human
has to upload a hero". There is no library and no search, so nothing returns it.

It is documented rather than deleted because a workflow that branches on 20 must
keep parsing rather than meet an unknown code. `NO_REFERENCE_EXIT` still works
and still remaps it. If you ever see a 20, treat it as a bug.

### How a finish status becomes an exit code

The model ends a run with one of four statuses. They are not interchangeable and
the exit code is derived from them, not from whether a file happens to exist:

| status | meaning | exit |
|---|---|---|
| `done` | the model judged the result good enough | 0 |
| `budget_exhausted` | the images ran out; the best of them shipped | 0 if `best.png` exists, else 1 |
| `gave_up` | the model judged nothing here shippable | 0 if `best.png` exists, else 1 |
| `no_candidates` | nothing was ever generated | **0 if the budget was 0, else 1** |

Two coercions are applied before the code is computed, so a status can never
claim more than the run delivered:

- **`done` with no candidates becomes `no_candidates`.** Otherwise a model that
  called `finish("done")` having generated nothing would exit 0 and ship
  nothing, which reads downstream as a successful delivery.
- **`no_candidates` splits on intent, not on `best.png`** - which cannot exist
  when nothing was generated, so testing for it would fail every such run
  including the ones that did exactly what they were asked. A run configured
  with a zero budget exits **0**; a run that had budget and still produced
  nothing exits **1**.

## `out/result.json`

Written every run, whatever happened:

```json
{
  "session": "20260827_143253",
  "outcome": "delivered",
  "exit_code": 0,
  "status": "done",
  "summary": "...",
  "best": "cand_04",
  "images_used": 3,
  "budget": 10,
  "attempts": ["cand_01", "cand_02", "cand_03"],
  "picks": 1,
  "images": ["best.png"],
  "reference": null,
  "grades": null
}
```

`outcome` is the field to route on - `delivered`, `delivered_at_budget`,
`nothing_shippable`, `no_candidates`, `error`. `exit_code` is for whoever is
debugging rather than routing. Kestra reds a task on any non-zero code, so a
flow that wants to handle "nothing shippable" as data rather than as a failure
should read `outcome`.

`reference` and `grades` are **retired but held as `null`** rather than dropped,
so a downstream parser that reads them keeps working. They carried the reference
receipt and the grader's scores; neither exists any more.

## What the image patches, and why

The harness source is byte-identical to the copy that runs natively on macOS.
The macOS-only assumptions are absorbed by the image instead, so nothing forks:

- **`sips`** - `docker/sips` reimplements the one flag form the tools use
  (`-Z <max> <src> --out <dst>`) on PIL, and exits non-zero on any other
  invocation rather than quietly doing something else.
- **`/System/Library/Fonts/Supplemental/Arial.ttf`** - symlinked to Liberation
  Sans, which is metric-compatible. Kept from the contact-sheet era; the tools
  that needed it are retired to `old/`, but the symlink is one line.
- **`magick`** - `docker/magick` is installed only when the base has ImageMagick
  6; it is inert on a base that ships IM7.

The build asserts that no `/Users/` path survives in `task/SKILL.md`. The skill
no longer hardcodes one, and the check is there so a path that creeps back in
fails the build rather than sending the agent looking outside the container.

`--yolo` is not optional and the entrypoint always passes it: `Approver.ok()`
denies every mutating tool when stdin is not a TTY, so without it the model is
refused on its first real step.
