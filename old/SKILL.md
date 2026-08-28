---
name: laydown
description: "Re-lay an off-set product photo so the garment sits square and wrinkle-free, keeping its real colour, texture and proportions. Write the prompt yourself from looking at the images, generate with fal.ai nano-banana-pro/edit, test what comes back, and deliver exactly 4 re-laid flats on a clean white plate, ranked best first."
---

# Laydown

## The goal

Deliver **exactly 4 images** of this garment, as re-laid flats on a clean white
plate, ranked best first.

Cutouts are currently off. `--ship` delivers the generated flats themselves;
pass `--cutout` if transparent-background PNGs are wanted again.

**Always 4, even when fewer than 4 are good.** These are options a person
chooses between downstream, not a finished delivery, and the caller needs a
fixed-size set. So the ranking carries your judgement instead of the length:
position 1 is the one you defend hardest, position 4 the one you would have
withheld.

That is a real cost and it is paid in `## Picking`. Every pick you would not
have shipped on its own must be named there with the defect it carries -
"generated_4 is cand_02, rejected for an added strap seam" - so a reviewer
reading top-down knows where your confidence stops. A named defect is a
warning; an unnamed one is a rubber stamp.

## The reference is already chosen. Do not choose one.

Before your first turn, the harness cleaned the source and then ran
`tools/select_reference.py`: it scored every image in `library_reference/`
against **the cleaned photo** - `<RUN_DIR>/archive/offset_upload.jpg`, tag
erased and background dropped, the same image you write the prompt from - took
the winner, desaturated it, and installed it at

```
inputs/reference_greyscale.jpg
```

That is the reference. There is exactly one, the inventory names it, and
`library_reference/` is deliberately absent from the inventory because it is not
an input - it is 45 photos of other garments. Do not go looking for it, do not
weigh a second candidate, and do not swap the reference for one you prefer.

Pass it to `prepare.py` explicitly and absolutely:

```bash
--reference /Users/ulmarti/Desktop/PLD_Harness/inputs/reference_greyscale.jpg
```

The receipt is `<RUN_DIR>/reference_selection.json` - which library file was
chosen, its score out of 100, the runner-up, a `differences` line naming what
still differs between the off-set garment and the reference, and `query` /
`query_cleaned` naming the image that was actually matched. If `query_cleaned`
is `false` the match was made against the raw photo with the tag and the room
still in it; that is worth a line in `## Notes`. **Read that
`differences` line before writing your prompt** and quote the source filename
and score in `## Setup`. It is the one place a real mismatch between the two
images is written down, and the checklist behind the score has no field for
most of what it catches.

The reference is greyscale on purpose - it is a shape and construction
reference, never a colour target. `prepare.py` says so again in the brief.

## Start with prepare.py. Do not go exploring first.

`tools/*.py` and `harness.py` are not inputs - the harness now refuses to read
them. This page plus `--help` is all you need.

**Do not list or read previous runs either.** They are not inputs and they are
not context. Three runs spent 44%, 54% and 51% of the context window on source
and old run folders before their first real step; one of them ran out of turns
after two images, with nothing picked and no log.

**Do not `ls` the inputs.** The workspace inventory at the top of this
conversation already names every input file with its dimensions and an md5, and
it is generated fresh each run. Read the filenames off it. Two runs in a row
burned turns on `ls -la inputs/` and `ls -la inputs/others/` for names that were
sitting in their own system prompt.

Your first tool call should be `prepare.py`.

## The tools

Run everything from `tools/` with the project interpreter:

```bash
cd /Users/ulmarti/Desktop/PLD_Harness/tools && ../.venv/bin/python <script> ...
```

| | |
|---|---|
| `prepare.py` | Checks the inputs, **verifies the pre-cleaned source** step 0 already produced at `archive/offset_upload.jpg` (it cleans only if step 0 did not), **inventories the construction**, and writes a prompt brief. Prints `RUN_DIR=` - carry it into everything else. Writes **no prompt**. |
| `stage_batch.py --run R --target 4` | **How to buy images.** Generates 4, grades them free, and buys more only if fewer than 4 cleared - the shortfall, never a cushion. Stops on the target, on the image ceiling, or on a round that bought images and cleared none. On a real batch this reaches the same four picks for 5 images (75c) instead of 10 (150c). `--ship` finishes with `--ship-faithful`. |
| `generate.py --run R --num N --resolution 2K` | The only billed step, and what `stage_batch.py` calls. $0.15 an image at 1K/2K, $0.30 at 4K. Numbering continues automatically, so topping up needs no extra arguments. **It refuses to spend on an unready prompt or a source the pre-clean gate rejected** - see `## The limits`. It appends the construction inventory and an image-2-is-layout-only clause to whatever you wrote; `--dry-run` prints the assembled prompt and bills nothing. |
| `grade_flats.py --run R` | **Grades and picks.** Measures fidelity to the cleaned source - silhouette, colour, texture - then checks with the vision model that the generation did not redraw the garment's construction. Prints a ranking and a `KEEP` list, and writes `archive/metrics.json` and `archive/grade_results.json`. It picks its own crop regions from the garment type in `reference_selection.json` and prints the line `profile: ...` - read it. `--profile bras`, `--profile leggings`, `--profile pullovers` or `--profile fleeces` overrides that. Stage-3 verdicts are judged once and reused on later passes; `--rejudge` re-rolls one deliberately. |
| `grade_flats.py --run R --expected-changes "..."` | Declares what the clean step legitimately removed - `"pearl-headed pins removed"` - so the judges do not report its absence as a defect. Use it whenever the cleaned source still shows paperwork, or whenever `## Results` would otherwise say every candidate altered every region. |
| `grade_flats.py --run R --ship-faithful 4` | **The delivery command.** Ships the 4 best that stage 3 found intact, backfills by grade only if fewer than 4 are intact, and prints every exclusion and backfill with the regions each carries. Use this one. |
| `grade_flats.py --run R --ship 4` | The blunt version: **top 4 by grade**, status ignored. Flagged regions are still printed per pick and written to `steps.log`. `--ship-clean-only` ships PASS only and ships fewer rather than backfill; `--cutout` adds transparent-background PNGs. |
| `crop_pair.py --run R --cand NN --at REGION` | Matching 1:1 crop boxes for two images that sit differently in frame. The region names follow the garment - `straps neckline cups centre band left right` for a bra, `waistband hip crotch thigh knee hem centre left right` for legwear, `collar neckline shoulders chest body hem centre left right` for a pullover (`left`/`right` are the sleeves, cuff included), the same set for a fleece plus `hood placket pockets` (`placket` is the zip line top to hem, `pockets` the hand-pocket band), plus `top upper middle lower bottom` for any of them. Run it without `--at` and it prints the set for this run; if the garment was never identified it withholds the garment-specific names instead of guessing. Image A defaults to `<run>/archive/offset_upload.jpg`, the cleaned upload. |
| `contact.py --run R` | Contact sheets of every candidate, cropped to the garment and sized to survive a vision call. Every crop box is checked against the source's, and a cell whose box is not credible says so rather than showing you a close-up of the wrong strip. |
| `compare_images` | The vision tool. Two images in one call. |

## The limits

- **One run means one folder, and one image budget.** `prepare.py` returns the
  same folder every time you call it - calling it again does not start over and
  does not reset the count. `generate.py` refuses to spend into any other
  folder. The ceiling is set by the operator, not by you; `--max-total` can
  lower it but never raise it. When the budget is spent, measure and pick from
  what you have. Four candidates is the floor for a four-image delivery, so if
  the budget ran out below that, ship what exists and say so in `## Picking` -
  it is the one case where fewer than 4 is unavoidable rather than chosen.
- **One prompt, written by you.** `prepare.py` leaves a brief in
  `archive/prompt_brief.md`. Write `archive/prompt.txt` with a bash heredoc.

  **Look at `<RUN_DIR>/archive/offset_upload.jpg`, never
  `inputs/off_set_image.jpg`.** The upload is the CLEANED image and the only one
  the model ever receives - tag erased, background dropped, plate white. The raw
  input still has the hang tag and a real-world background. A run wrote its
  prompt from the raw input, described the tag, asked that it "stay in place",
  and all four candidates grew a tag that was not in the image sent.

  ```
  compare_images(path_a="<RUN_DIR>/archive/offset_upload.jpg",
                 path_b="<the reference named in your workspace inventory>",
                 question="...")
  ```

  **The reference is `inputs/reference_greyscale.jpg`** - installed by step 0
  before your first turn, and named in the workspace inventory with its md5.
  Check the inventory rather than this page if the two ever disagree: it is
  fingerprinted and current. A run once guessed `reference_image.jpg` from an
  older copy of this page, found it missing, and spent two turns running `ls`
  over `inputs/` to discover a name it had already been given. `prepare.py`
  prints the reference it resolved - that line is authoritative.

  `generate.py` refuses a prompt under 120 words, one that never mentions
  greyscale, flatness, wrinkles or the background, one that asks for a
  **transparent** background (the model cannot output alpha and paints a
  checkerboard instead - ask for plain white, `cutout.py` adds transparency
  afterwards), and one that mentions a **tag, ticket, label, barcode or
  hanger** at all - there is none left to keep or remove, and naming it makes
  the model draw one. It prints every problem at once, so fixing them is one
  turn. `--force` sends it anyway and records that it did; use it only for
  something like a genuine sewn-in woven label that must survive.

  It also refuses a prompt that **names a viewpoint** - "shown from the back",
  "front view", "reverse side", "viewed from". That is how a garment comes back
  flipped: one run's prompt opened "the garment is shown from the back" and
  four of its ten candidates showed the other face, every seam correct. Say
  nothing about sides; `generate.py` appends the clause that keeps the face
  right.

  And it **warns** (does not refuse) on absolute smoothness - "wrinkle-free",
  "no creases", "freshly steamed". Ask for relaxed folds and kept texture
  instead; see clause 6 of the brief.
- **Never re-run to top up a failed image.** Everything on disk is already
  billed, and re-rolling one bad candidate with a tweaked prompt reliably makes
  it worse.
- **A source the pre-clean gate rejected is not generated from.** If step 0's
  outline gate found part of the garment missing from the cleaned image, the
  harness stops before your first turn (exit 21) and `generate.py` refuses to
  spend. This is not a formality: a run whose gate reported the bottom edge of
  the garment pulled in 16.7% carried on anyway and spent its entire 150c image
  budget on a source that had already been rejected. If you meet the refusal,
  look at `archive/clean_audit.json` and the cleaned image - do not reach for
  `--ignore-clean-gate` to get past it.

`generate.py` appends `archive/garment_description.md` to every prompt
automatically. **Never describe seams or panels yourself** - two vision models,
asked what this garment has, both invented a seam down each leg, and the
product's spec sheet says "No inseam". Your prompt needs one line saying the
construction is reproduced exactly as specified and nothing is added.

Where that inventory comes from is printed by `prepare.py` and recorded in
`steps.log`:

- **`inputs/Design_BOM.png` present** - transcribed from the spec sheet, both
  halves sent, authoritative. This is what you want.
- **absent** - inferred from the photo, and only the NOT-PRESENT half is sent.
  The positive half fabricated on all five attempts across two model tiers.

**The NOT-PRESENT half is audited before it is sent, and you should read what
it dropped.** Every item on that list becomes "the garment specifically does NOT
have this" in the prompt, so a false one tells the model to delete real
construction. A claim is withheld when the description's own Part 1 says the
thing IS there, when step 0 measured an attribute that says it is there, or
when no photograph could show it either way. On a real run that withheld
`Pearl embellishments` (Part 1 described four of them), `Side seams` (Part 1
located the pearls on them), and `Racerback straps` and `Pullover style` for a
garment step 0 had measured as exactly those - all four had been sent.
`describe.py` and `generate.py` both print the drops with reasons, and the
description file carries them in a comment.

The source is already clean by the time you write the prompt: measured on this
project, the hang tag and a clip were erased, the bench and shoes removed, the
plate turned pure white, garment colour drift **0.4**, full resolution kept. So
the re-lay prompt has only two jobs left - square the garment and de-wrinkle it.
Asking it to remove a background or a tag now invites it to invent one.

**Image 2 is a lay reference, never a construction reference.** Say that in your
own words, and never write anything like "it is a shape and construction
reference" - a real run's prompt said exactly that, and four of its ten
candidates came back with the reference's V-neckline seam and topstitching down
the straps, all four flagged by stage 3. `generate.py` appends a clause about it
to every prompt automatically, and when step 0 measured the reference as
differing in construction terms it names those words too. Read the
`construction_risk` line in `reference_selection.json` and the clause in
`archive/prompt_brief.md`: if it is flagged, that is the single most likely way
this batch wastes money.

You can read the full prompt - yours plus everything appended - without paying
for it: `generate.py --run R --dry-run` prints it and stops before anything is
uploaded.

## The tests

`grade_flats.py --run R` runs all of them and prints a `KEEP` list. It grades in
three stages and the third one is a door, not points on a scoreboard.

**Stage 1, measured against the cleaned source.** No model involved, so nothing
here can be invented. Every term is a comparison with `archive/offset_upload.jpg`
- the question is "is this still the same garment", not "is this a nice photo":

| | |
|---|---|
| `silh` | silhouette IoU against the source, both normalised to their own bounding boxes so re-centring and rescaling cost nothing. 0.95 scores 100, 0.70 scores 0. **35%** |
| `col` | dE76 between the two garment colours. 1.0 or less scores 100 (invisible), 6.0 scores 0 (a different colourway). **20%** |
| `wrink` | distance from the **source's own** texture energy, in either direction. Rougher means creases the re-lay failed to relax; smoother means the knit was ironed out of existence. **20%** |
| `sym` | the silhouette against its own mirror, 1.00 to 0.80. Presentation, not fidelity - a redraw is usually *more* symmetric than the real garment, which is why it is only **15%** |
| `bg` | backdrop lightness. `bg_lum` 0.99 scores 100, 0.90 scores 0 - anchored on the plate this pipeline actually produces, which sweeps 228-252 and never reaches pure white. Measured, not judged: the model rated a visibly grey backdrop 100/100. **10%** |

`grade = 35% silh + 20% col + 20% wrink + 15% sym + 10% bg`, minus **15 per
region stage 3 flagged as altered**, pass mark **62**.

That pass mark is not comparable to the 80 the old presentation grade used, and
neither are the numbers. This grades fidelity: on the batch it was calibrated
against, faithful re-lays scored 63-74 and redraws 40-48.

**Stage 2 is advisory and off by default.** Asking the vision model for 0-100
scores saturated at 100/100 for every candidate including one with a visibly
grey backdrop, and the pairwise tournament picked by slot 100% of the time.
Neither feeds the grade. `--judge absolute` runs one anyway, printed beside the
measurements.

**Stage 3, the construction gate.** Three native-resolution crops of each
candidate against the same crops of the cleaned source, one vision call each,
asking only whether stitching, seams, pockets, waistband or labels changed.
Which three regions depends on the garment, and `grade_flats.py` reads that off
`reference_selection.json` itself - the `profile:` line at the top of its output
says which set it used and where it got it. A line starting `WARNING:` there
means the garment could not be established at all: grading still runs so you can
look, but **`--ship` refuses outright** until `--profile bras`,
`--profile leggings`, `--profile pullovers` or `--profile fleeces` says which. Do not reach for `--no-construction` to get
past that - it does not lift the refusal, and it is the larger version of the
same mistake.

**A verdict is judged once and then reused.** Each candidate's stage-3 result is
stored in `archive/metrics.json` against a fingerprint of what was compared -
the candidate's bytes, the reference's bytes, and the crop bands - so grading
again, or grading again with `--ship`, reuses it and prints `(cached 21:32:42)`
instead of asking the model a second time. Only new images cost anything, and
the second pass shows `0 judged, N reused` in about 0s. This is why the flags in
your `## Results` table, the ones printed against each pick, and the ones in
`metrics.json` are the same flags: they are one judgement, not three samples of
a noisy one. If you want a second opinion, `--rejudge` takes one deliberately
and overwrites the stored verdict, so there is still exactly one record.
**Stage 3 also checks the FACE.** Each image is asked on its own whether it
shows the outside or the inside of the garment, and its front or its back; a
candidate whose answer differs from the source's is marked `FLIPPED`, costs the
same 15 points as an altered region, and is rejected. It is asked per image and
compared afterwards on purpose: the cheaper design - showing both crops and
asking "same face?" - was built first and measured **blind**, answering SAME for
a candidate mirrored left-for-right. A flip is invisible to every other test
here, because the silhouette, the colour and the texture are all unchanged.
Read the reason printed with each flag; the judge partly reasons from what
garments of that type usually look like, so a flag is meant to be appealable.

A **MISMATCH** marks the candidate REJECT **and costs it 15 points per altered
region**, but REJECT still does not block delivery - `--ship` takes the top N by
grade regardless. The penalty is there so the number and the label stop
contradicting each other: a run once had to arbitrate between a 79.1 grade and
three MISMATCH flags on the same image, and spent thirty turns on it. Every
MISMATCH that ships is printed against its pick - put those in `## Notes`.

**When every candidate is flagged the same way, suspect the check before the
batch.** Each candidate is an independent draw, so they fail in independent
ways: a real construction problem hits some of them, in different places, in
different words. A flag on *all* of them in the *same* region, worded almost
identically - "the band is missing", "no band visible" - is the signature of a
crop that is not looking at the garment, not of ten identical mistakes. The
usual cause is the wrong region set: a bra measured with the leggings bands puts
`waistband` on empty plate above the straps and `hem` below the garment, and two
empty crops compared against each other produce a confident verdict about
nothing. `grade_flats.py` prints this warning itself when it sees it. Work it in
this order:

1. Read the `profile:` line. Does it name the garment you are actually looking
   at, and did it *read* that or assume it?
2. **Look at the cleaned source for something the clean was supposed to remove
   and did not.** If a pin, clip or tag survived into `offset_upload.jpg`, every
   candidate correctly leaves it out and every region comes back MISMATCH for
   it. That is one defect in the source, reported ten times. Declare it:
   `grade_flats.py --run R --expected-changes "the three pearl-headed pins,
   removed"`, which re-judges (the flag is part of the verdict fingerprint).
   On the batch this was written for that took the flags from 30 of 30 down to
   12 of 30, and the twelve that remained were four candidates with genuinely
   invented topstitching.
3. Put one flagged pair in front of your own eyes -
   `crop_pair.py --run R --cand NN --at <the flagged region>` - and check the
   crop contains the thing the judge says changed.
4. Only then treat it as a per-candidate defect and appeal it image by image.

Appealing ten identical flags one at a time costs ten vision calls and confirms
nothing, because the fault they share is upstream of all of them.

Read the numbers honestly:

- **`wrink` is the weak one, and it is now a distance rather than a level.**
  `common.py` records that an isotropic variance measure of exactly this class
  was tried here as "lower is better" and removed: it ranked the visibly
  *smoothest* candidate highest because it was reading form shading rather than
  creases. Scored as distance from the source's own value the shading cancels,
  but it is still the term to distrust first. If the ranking disagrees with what
  you can see on `archive/grade_results.jpg`, say so in `## Notes` and pick past
  it.
- **The grade is a weighted average, so a good term can buy off a bad one.**
  Look at the `silh`/`col`/`wrink`/`sym`/`bg` columns and the `pen` column, not
  just the total.
- **The two starred columns (`sym*`, `smooth*`) are batch-relative and are not
  scored.** 100 there means "most of these", not "good". They are context for
  the case where the whole batch is weak - which the anchored grade will show
  as everything scoring badly, rather than as a winner.
- **Nothing here checks length, waistband width or frame clipping.** If a
  candidate looks stretched, or touches a frame edge, only you will catch it -
  `compare_images` against `archive/offset_upload.jpg`. Colour and outline ARE
  measured now (`col` and `silh`), so a desaturated or reshaped candidate should
  show up in the table before you see it.

**Do not re-generate to fix a failing candidate.** A repair pass rerolls the
dice rather than converging: one candidate scored 50 for integrity, was re-sent
with a corrective prompt, and came back at 40, having added texture that was
never there.

Framing, position, scale and tilt are **not tested and must not be**. The
retouch team places the garment themselves. Grading framing was
actively harmful - the candidates rejected for it carried the best colour and
texture in every batch, because they were the ones that left the product alone.

## Sequence it yourself

There is no prescribed order. Generate, grade, look, pick. Stop when you have
4, or when the cap is reached.

**Buy the images in stages.** `stage_batch.py --run R --target 4` does it:
four, grade, then only the shortfall. A batch is usually decided by its first
four - on the run this was measured against, five of the first six candidates
cleared and the other four were bought after the answer was already on disk.
Generating the whole budget in one wave is the single easiest way to spend
double for the same delivery.

**Look at every candidate, not just the top of the ranking.** One vision pass
per generated image, all of them, before you pick. The grade cannot see the
thing that matters most - whether the model redrew the garment instead of
re-laying it - and a redrawn garment often scores *well*, because clean
invented stitching reads as good texture. Runs that only inspected the top few
shipped a candidate with an added strap seam and another with a reshaped
waistband, both of which sat high in the ranking.

**Your eyes are the last say, over the numbers, in both directions.** Reject a
high scorer when you can see it is redrawn, and rescue a low scorer when you
can see the score is wrong - one candidate graded 65.6 on a flag that turned
out to be a false positive and was correct to ship. When they disagree, say so
in `## Picking` and say which you followed.

Three things that have gone wrong repeatedly, worth planning around:

- **Nothing cross-checks your picks any more.** `--ship-faithful N` applies its
  rule and writes the files, and that is the whole of it. There is no second
  opinion between the grade and the deliverable, so the numbers in `## Results`
  have to be ones you read off `grade_flats.py`'s own output rather than ones
  you remember. A run once reasoned its way to one set of picks and then typed
  the numbers out of an example, shipping two candidates that had been rejected.
- **Fewer than 4 cleared? `--ship-faithful 4` and let it backfill.** Do not
  generate more, and do not hand-copy files. It ships the candidates stage 3
  found intact first, backfills by grade only when there are too few, and
  prints every exclusion and every backfill with the regions each one altered.
  Then say in `## Picking` what the backfilled picks carry - it printed the
  lines, put them in the log. Rejected does not mean unusable, it means the
  cost is named. If the whole batch failed the same way, that is a prompt
  fault and more draws would only buy more of it; say that too, rather than
  letting four ranked images imply four acceptable ones.
- **Never end a run with `output/` empty.** If the turns run out first, the
  harness ships the four most faithful candidates itself and says so - but a
  delivery it wrote carries no `## Picking`, no `## Notes` and no account of
  what each pick costs, which is most of the value. Three turns from the cap
  you will get a message saying exactly this; deliver then, do not keep
  investigating. A real run reached its cap mid-appeal with ten paid-for
  images in `archive/` and nothing delivered.
- **A construction MISMATCH now ships anyway, so it has to be reported.** It is
  the only check in the pipeline that can tell a re-laid garment from a redrawn
  one, and it runs on 1:1 crops precisely so it is not guessing - but it no
  longer stops anything. `## Notes` is the only place that record survives, so
  name every flagged pick and the regions it altered. `crop_pair.py --run R
  --cand NN --at <region>` puts the two crops in front of you if you want to
  judge a flag yourself before writing it up - use the region `grade_flats.py`
  named, and run it without `--at` first if you want the list for this garment.

## Deliver

`--ship-faithful N` writes the picks to `output/`. Then write `<RUN_DIR>/LOG.md`
with a **bash heredoc**, not `write_file` - a long string argument truncates
mid-JSON and the call is rejected.

Sections: `## Setup` `## Prompt` `## Generation` `## Testing` `## Picking`
`## Results` `## Notes`

`## Setup` names the reference step 0 installed - the library filename, its
score, the `differences` line, and whether `construction_risk` was flagged -
read off `<RUN_DIR>/reference_selection.json`, not from memory.

`## Results` is a table: Pick | File | Grade | Silhouette | Colour | Wrinkle |
Background | Construction, with the source's own wrinkle energy and symmetry
quoted underneath, since every one of those columns is a comparison against it.

`## Notes` carries the honest caveats - a speck in a background, a candidate you
nearly picked, any `--force` and why, and whether any pick is a no-op.

**Then call `finish()` - the same turn, nothing in between.** LOG.md is the last
artefact; once the heredoc has written it the run is over. Nothing after it adds
anything, and the two things that usually fill the gap both cost the run:
re-reading your own outputs to check work already recorded, and one more look at
a candidate you have already ranked. Keep going and the iteration cap arrives
instead, which ends the run with no `finish()` at all - the picks sit in
`output/` either way, but nothing says they were the picks, and an unfinished
run reads as an abandoned one. Summarise what shipped, name the flagged picks,
and stop.

## Rules that cost real runs

1. **Pass every path explicitly and absolutely.** A script falling back to a
   default input will process a different garment and report precise, plausible,
   entirely wrong numbers.
2. **Ground every number in output you actually saw.** Never report a
   measurement you did not run or a file you did not create.
3. **`cat <RUN_DIR>/steps.log`** is the whole run in a few lines, and the spend
   record to quote. Read it instead of re-deriving.
4. **Costs are calculated at published rates, never receipts.** fal exposes no
   billing API.
5. **Keep tool output small.** You have a limited context window and a run that
   fills it ends before the work does.
