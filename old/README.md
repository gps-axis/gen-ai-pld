# Retired: the reference-matching and grading pipeline

Nothing in here is imported, called or shipped. It is kept because it encodes a
lot of measured behaviour that a future change might want to read, and because
the archaeology comments in `grade_flats.py` and `common.py` are worth more than
the code around them. Git history has everything anyway - this folder is just so
you do not have to go looking.

## Why it went

The harness used to be a fixed five-stage assembly line: clean the photo, search
a 45-image library for a reference, describe the garment's construction, write
one prompt, buy ten images in a single wave, grade them on a weighted five-term
score behind a three-stage vision gate, and ship exactly four ranked picks.

The model's only real decision in all of that was which four to name at the end.
It could not look at what came back and try again, could not change one line of
the prompt, and could not clean up a generated image. The project now does one
thing - iterate an off-set photo towards a reference laydown, with the model
driving - so the machinery that made the decisions for it had no job left.

## What is here

| | |
|---|---|
| `tools/select_reference.py` `tools/match_reference.py` | step 0 - scored every image in `library_reference/` against the cleaned photo and installed the winner. The reference is now supplied by the operator |
| `tools/grade_flats.py` | the five-term score, the three-stage vision gate, the face check, and the shipping logic. Replaced by the model looking, plus the three advisory numbers in `tools/metrics.py` |
| `tools/prepare.py` `tools/describe.py` | input validation and the VLM construction inventory that got appended to every prompt |
| `tools/stage_batch.py` | the cost-staged buy-four-then-top-up loop. The model now decides when to spend |
| `tools/clean.py` | the fal-based pre-clean. Its endpoint `fal-ai/image-editing/object-removal` has answered 403 since 2026-08-27, which is why `tools/segment.py` exists |
| `tools/measure.py` `tools/review.py` `tools/contact.py` `tools/crop_pair.py` `tools/cutout.py` | superseded by `grade_flats.py` before this revamp, or orphaned by it |
| `tools/vision.py` | the Qwen vision client. The harness talks to the same server directly, so nothing imports this any more |
| `tools/profile_library.py` `profiles/` | offline layout profiling. Its output was read by nothing |
| `library_reference/` | 45 finished PDP photos, the search corpus for step 0 |
| `notes/AGENTIC_PLAN.md` | a 2026-08 proposal, already stale when this revamp started |
| `SKILL.md` | the old operating manual. The lessons in it that still apply were carried into `task/SKILL.md`; the rest described machinery that no longer exists |

## If you bring something back

`common.py` and `runlog.py` stayed in `tools/` and are unchanged, so most of
this will still import. The things that will not resolve are
`reference_selection.json` (nothing writes it now), `archive/offset_upload.jpg`
(now `archive/source_clean.jpg`), and `archive/garment_description.md`.
