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

`out/best.png` is rank 1 of the delivery, with `best_2.png` to `best_4.png`
beside it. `out/result.json` is the receipt, `out/used_prompt.txt` the prompt
rank 1 was generated with. `out/logs/` gets `steps.log`, `LOG.md`, `run.log`,
`transcript.jsonl`, the run's `lineage.json`, `prompt_sections.json`,
`notes.json`, `picks.json`, `metrics.json`, and the two images the model was
actually working from (`source_clean.jpg`, `reference_greyscale.jpg` and
`reference_original.jpg`). All of it is written by `tools/deliver.py`, which
runs by hand too.

## Under the Kestra flows

The flows in `kestra/` were written against the retired pipeline in `old/`, and
Axis reads their webhook payloads, so the container keeps that contract rather
than the flows changing. The entrypoint translates:

| the flow sends | what happens |
|---|---|
| `--reference-category CAT` | only `<library>/CAT` is searched; a missing or empty folder falls back to the whole library |
| `--no-reference-select` | the file at `/app/inputs/reference_greyscale.jpg` is the reference, no search |
| `--task` naming `grade_flats`, `stage_batch` or `--ship` | dropped with a warning; the harness delivers four on its own |
| `-e OUTPUT_PATTERN=generated_{n}.png` | the ranked picks land in `OUT_DIR` under that name |
| no `REFERENCE_LIBRARY`, no `/in/reference_library` | `/app/library_reference`, where the flows mount the shared library |

And `deliver.py` writes the files the flows read back: `used_prompt.txt`,
`result_top_matches.jpg` and `match_results.json` (the reference near-miss strip
and selection record under their old names, in `OUT_DIR` and in the run
folder), `output/pickN_cand_XX.png` copies of the picks, and
`archive/metrics.json` with one row per candidate whose `score` is the lay
match to the reference as a percentage.

Two things the flows need that the old image did not: `SEGMENT_URL` and
`SEGMENT_API_KEY` for the segmenter. Without the key the run continues from
the raw photo and says so in `run.log`.

## The reference: found, or supplied

The reference laydown is chosen automatically from a library, and you can
override that by supplying one.

**Mount a library** at `/in/reference_library` (or point `REFERENCE_LIBRARY` at
it) and the harness picks the reference itself: it describes the garment, scores
every library image against it, and confirms the winner with the model. Nothing
is billed by any of that. See "Choosing the reference" below.

**Or supply one** and no search happens. The entrypoint takes whatever matches
`/in/reference*`, or name it explicitly with `-e REFERENCE=/in/my_reference.jpg`.
An explicit reference always wins over the library.

The garment photo is either the path you pass as the first argument, or - if you
pass none - the single image in `/in` that is *not* named `reference*` and is not
inside the library folder. It refuses to guess between several: choosing the
wrong one costs a full run at fal.ai.

With neither a library nor a reference, the run stops before it starts and says
so.

Anything after the image path goes straight to `harness.py`:

    docker run ... pld-harness /in/photo.jpg --max-iters 60 --max-images 4

## Choosing the reference

When a library is mounted, step 0 picks the reference before the agent starts and
before anything is billed:

1. **Describe.** One model call fills in a fixed form for the garment and for
   each library image - body region, sleeve, leg, length, silhouette, front
   opening, neckline, waist, hem, fabric, structure. Cached on content under
   `/app/.cache/refmatch`, so a library is described once.
2. **Score.** Plain arithmetic, no model. Anything that cannot work at all is
   disqualified outright - a different body region, sleeves against none, legs
   against none, a front that opens all the way against one that does not - and
   the rest are weighted into a number out of 100.
3. **Confirm.** One call showing the garment and the survivors side by side. The
   model picks one or answers "none". Both gates must pass: the score clears the
   threshold *and* the model does not reject it. Otherwise exit 20.

`runs/<session>/reference_match.jpg` shows the garment beside the top candidates
with their scores, and is copied to `/out/logs/`.

**The threshold is provisional.** It defaults to 78, which was measured on nine
of this project's own assets, not on your library. Set it from your own data:

    docker run ... --entrypoint /app/.venv/bin/python pld-harness \
      /app/tools/select_reference.py --library /in/reference_library --calibrate

That scores every library image against every other and prints where same-garment
pairs sit against different-garment ones. Pass the number you read off it as
`--reference-threshold`. Use `--index` the same way after adding images, so a
live run never pays to describe them.

## The container is a client, not a system

Nothing is bundled but the code. Three services stay outside it, and each has to
be reachable from inside the container:

| what | env var | default |
|---|---|---|
| text/vision model (the agent) | `QWEN_BASE_URL` | `http://10.11.245.41:8091` |
| which model, when the endpoint serves several | `QWEN_MODEL` | none - the first model `/v1/models` lists |
| SAM3 segmentation service | `SEGMENT_URL` | `http://10.11.245.145:4000/sam3-segment` |
| its key, separate from the model's | `SEGMENT_API_KEY` | none - unset means no auth header |
| fal.ai (the billed generation) | `FAL_KEY` | none - required unless the budget is 0 |

`QWEN_API_KEY` is required. On the host it comes from `.qwen_key`; that file and
`.env` are in `.dockerignore` on purpose, because anyone who can pull an image
can read its layers. Pass them with `-e`.

`REFMATCH_BASE_URL` is not read by anything. The reference search is back, but
it talks to `QWEN_BASE_URL` like everything else - there was a second vision box
when that variable existed, and two names for one endpoint meant one of them was
always stale. Passing it is harmless and ignored.

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

`-v pld-cache:/app/.cache` is worth having again. The reference search caches one
description per library image under `.cache/refmatch/`, keyed on content, and
without a persistent cache every container re-describes the whole library on
every run - one model call per image. Nothing is billed either way; it is time.

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
| 2 | misconfigured: missing credential, no image, no reference and no library, several images |
| 3 | the harness reported success but nothing reached `/out` |
| 20 | nothing in the library is close enough to this garment - see below |
| 21 | the segmenter returned an image with most of the garment missing |

### 20: no reference could be found

The library was searched and holds nothing that can serve as this garment's
laydown. Not a fault, and not a crash: the run stops before the agent starts and
before a single billed image, and `runs/<session>/reference_selection.json`
carries `match_found: false` along with the closest miss and its score.
`reference_match.jpg` shows the near misses side by side, which is usually the
argument for adding one asset rather than shooting a new garment.

The fix is a human supplying `-e REFERENCE=...`, or adding a suitable laydown to
the library. `NO_REFERENCE_EXIT` remaps the code for a caller that needs a
different one.

This was documented as unreachable for a while, when the reference was
operator-supplied and there was no search left to fail. It can occur again.

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
  "picks": 4,
  "ranked": [{"rank": 1, "candidate": "cand_10", "chosen_by": "model"}, "..."],
  "images": ["best.png", "best_2.png", "best_3.png", "best_4.png"],
  "reference": null,
  "grades": null
}
```

`outcome` is the field to route on - `delivered`, `delivered_at_budget`,
`nothing_shippable`, `no_candidates`, `error`. `exit_code` is for whoever is
debugging rather than routing. Kestra reds a task on any non-zero code, so a
flow that wants to handle "nothing shippable" as data rather than as a failure
should read `outcome`.

`reference` carries the selection receipt again when the library was searched -
which library image won, what it scored, and what the model said still differs.
It stays `null` when the reference was supplied by hand, because there was no
choice to record. The full record is always at
`runs/<session>/reference_selection.json`.

`grades` is **retired but held as `null`** rather than dropped, so a downstream
parser that reads it keeps working. It carried the candidate grader's scores; that
step does not exist any more.

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
