---
name: laydown
description: "Re-lay an off-set product photo into the reference laydown: the same silhouette, ironed flat, the same size in the frame, keeping the garment's real construction and colour. Deliver the best FOUR candidates, ranked. Write the prompt yourself, generate with fal.ai, look at what comes back, and keep the four best until the images run out."
---

# Laydown

## The goal

Image 1 is a real garment photographed off-set - pinned to a wall, creased,
shadowed. Image 2 is what a finished ecommerce laydown looks like: square, flat,
sleeves symmetric, clean white plate, no shadows.

Get image 1 into the lay of image 2. **In this order:**

1. **The silhouette matches image 2** - the same shape, the sleeves or legs at
   the same angle and spacing, square and level.
2. **Ironed flat** - every crease and handling fold gone, the fabric smooth.
3. **The same size in the frame as image 2**, with the same margins.
4. **The construction is image 1's** - every seam, pocket, waistband, cuff,
   sewn-in label and logo. This is a gate, not a score: a candidate that hit
   the lay by inventing a seam or losing the logo is not shippable.
5. **Colour** is second-level. A flat, evenly lit render reads lighter than a
   creased photograph, and that is not a fault: judge colour on the **hue**
   part of the reading. Under 4 is the same colourway. Over 8 is a different
   one - a blue where the garment is green - and that disqualifies, however
   good the lay. Between the two, the lay decides.

Image 2 is a **different product**. Its collar, its trim and its pockets are
not yours. Take the lay, the flatness and the size, and nothing else.

**The delivery is the best FOUR candidates, ranked**, not one. `pick_best`
takes a list, best first, and only UNTOUCHED generations - `cand_NN` exactly
as fal.ai returned it. A segmented or polished form is refused; generate from
it and name what comes back. Once you have one good lay, do not
keep refining it: spend the rest of the budget on new seeds and collect
alternates. Whatever looks best in this run ships; there is no minimum, and
the harness fills any slot you leave empty from its own lay ranking and says
so.

Why this order is written down: `runs/20260902_100812` shipped a candidate at
0.892 against the reference silhouette, with good colour, while two candidates
at 0.96 - flat, sized like the reference - were passed over for being pale.

## The loop

There is no fixed order. Every tool is available every turn.

1. Look at both images. Say what actually differs - pose, creases, shadows,
   anything still attached to the garment.
2. Write the prompt with `prompt_set`, section by section. Check it with
   `prompt_show`. Both free.
3. `generate` one or two images. Look at what comes back.
4. Decide: fix one prompt section and generate again from `source`, or take the
   best candidate and generate from **that** to fix what is left.
5. `pick_best` with your ranked list as soon as anything is worth shipping.
   Re-issue the whole list as better ones arrive - the last call wins.
6. `finish` when it is good enough, or when the images run out.

`segment` works on candidates too, and the result (`cand_03s`) is a name every
other tool accepts - but it is a CUTOUT: the garment on flat white, the plate
and its shadow gone. Measure it, or generate from it so a plate comes back. `pick_best`
refuses it: a cutout shipped as `best.png` on `runs/20260902_233202`, before
the refusal, and was rejected for exactly that look. A candidate that came back with a hanger
hook or a dirty plate is a bad draw: name a clean sibling, or generate from
the cutout.

**What `segment` can and cannot rescue.** It drops the BACKGROUND: a grey plate,
a soft shadow, specks. It does not choose between things standing on that
background, so it cannot remove a **second garment** - measured on
`runs/20260827_220727/archive/cand_05.png`, which came back with a duplicate
sweater stacked behind the real one: segmenting it cleaned the plate perfectly,
kept both sweaters, and tore a sleeve off the upper one. A duplicate is a bad
draw, and the fix is a new seed, not a cleanup.

## Look at the segmented source, never the raw photo

`<RUN_DIR>/archive/source_clean.jpg` is the image fal.ai actually receives, and
it is the one pinned as image 1 in your first message. `inputs/off_set_image.jpg`
is the raw photo with the room, the wall and the shadow still in it.

**Write the prompt from the segmented one.** Anything you describe from the raw
file is something the generator was never sent - and describing it is how it gets
drawn. A run wrote its prompt from the raw input, described the hang tag, asked
that it "stay in place", and all four candidates came back wearing a tag that was
not in the image sent. A test run of this loop reached for
`inputs/off_set_image.jpg` on its second turn and started inspecting a thread
loop that segmentation had already removed.

The exact paths are printed in your first message. Use those.

## Do not go exploring first

Your first call should be `compare_images` on the two images you were given, and
your second should be `prompt_set`.

**Do not list the workspace, read previous runs, or grep the harness source.**
None of it is an input. Three runs spent 44%, 54% and 51% of their context window
on the source tree and old run folders before taking a single real step; one ran
out of turns after two images with nothing picked. A zero-budget test run of this
very loop still burned two turns on `ls runs/` and `grep budget harness.py` -
both answers were already in its own system prompt.

Everything you need is in this page, in the two images, and in `prompt_show`.
`<script> --help` covers the rest.

## The budget is the only scarce thing

Images are capped for the whole run and the counter never refills. Everything
else - looking, comparing, measuring, segmenting, rewriting the prompt - is free
and unlimited, so there is never a reason to skip a look to save something.

**Buy one or two at a time.** A batch spent in one wave cannot learn from itself.
A run that generates all ten up front has made one decision; a run that generates
two, looks, fixes a section, and generates two more has made three.

**Separate a prompt problem from a bad draw, because the fix is different.**

A **prompt problem** shows up in most or all of a batch, in the same way. Fix the
section that governs it. Re-rolling will not help - the next draw obeys the same
wrong instruction.

A **bad draw** hits one candidate while its siblings, from the identical prompt,
are fine. The commonest by far on this project is a **second garment in the
frame** - a faint or stacked duplicate of the same sweater, which has run between
2 and 8 in 10 on past batches. A stray shadow or one badly-posed sleeve can be
the same thing. No wording fixes these; the appended clause already says the
frame holds one garment, and saying it again makes it worse, because naming the
thing is how a diffusion model is told to draw it.

**Change the seed instead.** Same prompt plus same seed is the same picture, so a
new seed samples somewhere else entirely:

```
generate(source="source", num=2, seed=500)
```

The seed sets the base for the wave, so that gives you 500 and 501. Omit it and
numbering just continues. When a batch is mostly right and one candidate is
carrying a duplicate, a fresh seed on the *unchanged* prompt is the cheapest move
available - you keep everything the prompt already got right.

**What re-rolling will NOT fix** is a candidate the generator redrew: sending it
back with a corrective prompt rerolls the dice on everything. One candidate
scored badly, was re-sent with a correction, and came back worse, having added
texture that was never there.

## Writing the prompt

`prompt_show` lists six seeded sections and what belongs in each. Fill the ones
that apply, add your own if the garment needs it, and edit **one section** when
one thing is wrong - rewriting everything re-rolls every decision that was
already right.

Some standing clauses are appended automatically to every prompt: that the frame
holds one garment, that image 2 is a lay reference only, that the lay is square
and symmetric with the hem level, that the same face must show, and the
pins-and-labels rule below. Do not write your own versions of these; you will be
arguing with them.

### Pins come off, sewn-in things stay - automatically

Every prompt already says that any **pin, clip, tack, hanger, hook, price ticket
or swing tag** holding the garment up for the photograph is gone from the
finished picture, and that the fabric it was holding lies flat and closed. It
also says that **everything sewn into the garment stays exactly as it is** -
seams, stitch lines, sewn-in brand and care labels, embroidery, appliqué, and
printed, woven or knitted logos.

That distinction is the whole point: what was holding the garment up is
temporary and goes, what is sewn into it is the product and stays.

It is standing because it has to be. Segmentation drops the **background only**,
so a pin or a hang tag attached to the garment survives into image 1 every time
and nothing else removes it. On `runs/20260827_223611` not one prompt section
mentioned them, and they shipped.

**So do not write your own.** `prompt_show` warns you if you do. Two sets of
words about "labels" is how the sewn-in ones get stripped along with the
temporary ones, and naming a thing repeatedly is how a diffusion model is told to
draw it - a prompt that described a hang tag and asked it to *stay in place* got
four candidates that grew one they had never been sent. Add a sentence only for
something genuinely unusual that the standing clause cannot know about.

**And it is checked, both ways.** Your first message lists what is actually on
the source - the fastenings that must go and the sewn-in detail that must stay -
read off image 1 automatically. Then every candidate arrives with a **PINS AND
LABELS** paragraph doing the same read on the result. So you can compare rather
than guess:

> SOURCE: a small pin/tack at the very top centre of the fur collar, and tiny
> tacks at the lower hem near both side seams. The white brand label inside the
> collar is present and intact.
>
> CANDIDATE: no pins, clips, tacks, hangers, tickets or string anywhere. The
> ribbed cuffs, hem, placket and four front buttons are intact.

Both halves matter. A candidate that lost the brand label or the embroidery along
with the pins has failed the same instruction from the other side, and a clean
garment missing its logo would otherwise read as a success. When the check says
something is still there, or something sewn-in has gone, do not `pick_best` it.

### The sleeve angle is yours, and nothing else sets it

The standing clause fixes symmetry, level shoulders and a level hem. It says
**nothing about the angle of the sleeves**, and that is deliberate: the reference
is whatever the operator supplied, and it can pose them any way.

So look at image 2 and write the angle into `pose` in your own words - splayed
wide away from the body, angled down at roughly 45 degrees, hanging straight at
the sides, cuffs level with the hem or above it. Be specific about how far the
cuffs sit from the body, because that is the single most visible thing about a
laydown and the thing a person notices first when it is wrong.

**This is the most expensive mistake made on this project so far.** On
`runs/20260827_215408` the reference had the sleeves splayed wide at about 35
degrees. The clause used to read "sleeves straight down at the sides", the agent
wrote its `pose` section to agree with the clause rather than with the picture -
"both sleeves brought in from their splayed position, close to the body" - and
all eight candidates, 120 cents' worth, came back with the sleeves tucked in.
Everything else about them was right.

If your `pose` never mentions sleeves, `prompt_show` warns you. Do not generate
past that warning.

### When the words fail: `match_pose`

Words have a ceiling. On `runs/20260827_222231` the `pose` section read "both
sleeves splayed wide away from the body, angled down at roughly 30 degrees,
cuffs well clear of the body" - a correct description of the reference - and the
sleeves still came back tucked in. A sleeve angle is a geometric fact about a
picture, and describing a picture in words loses something that pointing at it
does not.

So when you have already described the pose correctly and been ignored:

```
generate(source="source", num=2, match_pose=True)
```

That takes the sleeve angle, the cuff spacing and the hem line off image 2
directly. It composes with everything else - `source="cand_03"` sends your best
candidate as image 1 and the reference as image 2, which is the move when the
construction is already right and only the arms are wrong.

**It is off by default because it has a real cost.** Pointing at image 2 for pose
is what drove the second-garment rate to 60-80% on three past runs, against
20-44% without it: told to reproduce image 2, the model reproduces image 2, and
image 2 contains a garment. Buy one or two, look at them, and if a duplicate
turns up **change the seed** - the prompt is not what is wrong. Do not switch
`match_pose` back off because a duplicate appeared; you would be trading the
problem you fixed for the one you started with.

**Three things are refused outright**, because each has a measured failure behind
it:

| | |
|---|---|
| naming which **face** is shown - "from the back", "front view", "reverse side" | one prompt opened "the garment is shown from the back" and seven of ten candidates came back inside-out, every seam correct |
| asking for a **transparent** background | the endpoint cannot output alpha and paints a literal checkerboard into the pixels. Ask for plain white |
| **"remove the background"** | it is already gone. Asking invites a new one |

Overhead camera wording is fine and is not a viewpoint in this sense - "flat lay
viewed from directly above", "top-down" all pass.

**Warned, not blocked:**

- **Asking to keep the creases, folds, texture or proportions.** The finished
  flat is ironed smooth and shaped like image 2; what survives is the
  construction, the colour and the pattern. Say "lies completely flat and
  smooth" and name the seams, pockets and labels that stay.
- **Naming a tag, pin, clip or label - it is already handled, so do not.** Every
  prompt carries a standing clause covering this, and your own version fights it.
  See below.

## The numbers

They arrive automatically under each new candidate, **in the order they
matter**, and all against the **original** source and the run's reference,
never a candidate's parent.

```
cand_10  LAY 0.969 vs ref (source starts 0.867)  size x0.99  flat x0.39  colour dE 24.1 (hue 4.2)  same-garment IoU 0.863
```

**LAY - the primary number.** The candidate's silhouette against image 2,
bbox-normalised. The source's own number is printed beside it so you can read
movement: on `runs/20260902_100812` the source started at 0.867, the shipped
candidate reached 0.892 (it barely moved), and the two best lays hit 0.959 and
0.969. Below about 0.90 the pose is still the source's. The pose section, or
`match_pose`, is what moves it.

**size** - the garment's share of the frame over image 2's. 1.0 is right. A
candidate at 1.4 kept the source's framing; a redraw usually lands near 1.0.

**flat** - surface energy over the source's. **Below 1 is wanted now.** A flat
redraw sits at 0.4-0.5; a re-lay that kept the creases sits near or above 1.
Far below 1 is only a problem if the construction went with the creases - check
the PINS AND LABELS read, not the number.

**colour dE (hue)** - read off the lit fabric of both garments, so a creased
photo and a flat candidate compare as fairly as they can. Even so, a flat
render reads several L* lighter than the photograph, so the whole dE runs
5-10 on a good candidate. **Second-level, and judged on the `hue` part**:
under 4 is the same colourway; over about 8 is a different one and
disqualifies; between, the lay decides.

**same-garment IoU** - the candidate against the source silhouette. It **falls
when the lay changes**, which is the job, so a high number here with a low LAY
means the candidate stayed where it was. Below about 0.58 you are not looking at
the same object at all.

**Specks** is a good tiebreaker when two candidates look alike.

`pick_best` tells you when a candidate outside your list sits closer to the
reference's lay than one inside it. If you are passing it over, say what
disqualifies it - construction, a second garment, a hanger - in a `note`.

## The lay check - read it, it is the one that catches the pose

Every new candidate arrives with a **LAY vs the reference** paragraph under it,
alongside the three numbers. It is a vision comparison of that candidate against
image 2, asked automatically and for free, about the **pose and nothing else** -
sleeve angle, how far the cuffs sit from the sides, whether the shoulders and hem
are level and the garment is square.

It reads like this:

> In the reference the sleeves drop almost straight down, angled only about
> 15-20° outward, with the cuffs ending near the side seams at the hem line. In
> the first image the sleeves are folded diagonally inward at a steeper 30-40°
> angle, with the cuffs tucked up high, well inside the body sides. **They do not
> match: sleeves too far in and too high.**

**This is the check the numbers cannot do.** IoU, dE and the wrinkle ratio all
answer "is this still the same garment". None of them answers "did I hit the pose
I was aiming at", and that is the question the reference exists for.

It is automatic because it did not happen otherwise: on `runs/20260828_110544`
the entire run contained two comparisons - source against reference to write the
prompt, and source against a candidate to check fidelity - and **not one**
comparing a candidate to the reference. Every candidate was checked for being the
right garment and none for being the right shape, and the run shipped with the
sleeves wrong.

**When it says they do not match, act on it.** Fix the `pose` section to say what
it just told you, or reach for `match_pose=True` if you have already described it
correctly. Do not `pick_best` a candidate whose lay check says the sleeves are
wrong just because its numbers look good - they are answering a different
question.

### A `!` line means the numbers are not evidence

Any candidate printed with a `!` under it has a measurement problem, and the
numbers on that line are noise rather than a bad score. **Do not quote them, do
not rank on them, and do not put them in your rationale.** Look at the picture.

The one that will bite you most is a **pale garment on a pale plate**. Cream on
white gives chroma no colour to separate and luminance no contrast, so both cues
collapse and the outline is simply not found. On `runs/20260827_220727` every one
of six candidates measured this way - the mask covered 2.8% to 4.2% of frames the
sweater filled - and the run shipped quoting "IoU 0.325" as if it meant
something. It meant nothing.

The opposite `!` line - the mask covering far MORE than the source - is worth
acting on rather than ignoring: it usually means there is a **second garment in
the frame**. Go and look, and if there is, change the seed.

## Editing an edit

Generating from a candidate is how you fix one remaining defect without losing
what already worked. It also compounds drift: image 1 is then a generated image,
so anything the model invented on the first pass arrives on the second labelled
as the real product.

`lineage.json` records the depth and it is printed with each candidate. **Two
edits deep is where invented detail starts to harden.** At depth 2 or more, look
at the candidate against the *original* source with `compare_images`, not against
its parent, and check the trim and the seams specifically.

Measured on this project, one chain, every number against the original source:

| | dE | what happened |
|---|---|---|
| `cand_02` (depth 1) | **1.0** | one generation from the real photo - invisible drift |
| `cand_02s` (depth 1) | **0.9** | segmented. No drift, because segmentation cannot add any |
| `cand_03` (depth 2) | **2.3** | one more generation. Drift more than doubled in a single hop |

Nothing there is broken - 2.3 is still the same colourway - but it is the shape
of the cost, and it compounds. A second edit buys one fix and pays for it in
fidelity, so spend it on a defect you can actually name.

Segmenting a candidate does **not** add depth. It drops a background; it cannot
invent anything. It can add background specks, though - `cand_02s` went from 41
to 276 - so look at the plate afterwards rather than assuming it improved.

## Look at every candidate

Not just the ones that score well. A redrawn garment often looks *better* than a
re-laid one, because clean invented stitching reads as good texture. Two runs
that only inspected the top of their ranking shipped a candidate with an added
strap seam and another with a reshaped waistband.

`compare_images` puts two frames in one vision call. Use it whenever the question
is how one differs from another - two separate `view_image` calls give you two
independent descriptions, and the gap between two descriptions is not a measured
difference. For a close look at a seam or a cuff, pass a box in source pixels;
without one the whole frame is squeezed to 1024px and you see nothing of the kind.

## What ships: untouched generations only

The four delivered files are `cand_NN` exactly as fal.ai returned them. There
is no polish pass and no recolour pass, and `pick_best` refuses a segmented or
polished name. A cutout (`cand_03s`) is for measuring or for generating from;
if a candidate needs its plate cleaned, generate from its cutout and deliver
what comes back.

## Finishing

`pick_best` names what ships: up to four, best first. Call it early and
re-issue the list as it changes - it costs nothing, and it means a run that gets
cut off still delivers what you chose rather than whatever happened to be last.
The four land in `output/` as `best.png`, `best_2.png`, `best_3.png` and
`best_4.png`, with `picks.json` beside them saying which were yours and which
the harness filled in.

Then `finish` with an honest status:

| status | means |
|---|---|
| `done` | the result is good enough |
| `budget_exhausted` | the images ran out; shipping the best of them |
| `gave_up` | nothing here is worth shipping |
| `no_candidates` | nothing was ever generated |

Write `<RUN_DIR>/LOG.md` first, with a **bash heredoc** rather than `write_file` -
a long string argument truncates mid-JSON and the call is rejected. Say what you
changed each round, what each generation started from, and what the winner still
carries. A named defect is a warning; an unnamed one is a rubber stamp.

## Rules that cost real runs

1. **Ground every number in output you actually saw.** Never report a
   measurement you did not run or a file you did not create.
2. **`cat <RUN_DIR>/steps.log`** is the whole run in a few lines, including the
   spend. Read it instead of re-deriving it.
3. **Costs are published rates, never receipts.** fal exposes no billing API.
4. **Keep tool output small.** The context window is the second scarcest thing
   here, and a run that fills it ends before the work does.
