# Plan: make the laydown flow agentic

Status: **not applied.** `tools/crop_pair.py` is done and verified; everything
below is proposed.

## The problem

The workflow is five scripts in a fixed order. The agent's only real decision is
which numbers to pass to `review.py --picks`. Replace it with a bash script that
takes the top 4 by score and the output barely changes.

The judgement is frozen into constants I chose at authoring time:

| Frozen today | Where | Why it should move |
|---|---|---|
| The whole prompt text | `prepare.py` `PROMPT` | Written without seeing the images |
| Garment silhouette wording | `prepare.py` `SHAPES` | Asserts what leggings look like from a lookup table. It described a bralette while the input was leggings |
| `--reject-tilt 3.0`, `--reject-colour 20.0` | `measure.py` | Guesses, applied as hard rules |
| Score weights `-3.0 / -40 / -0.5 / -0.3` | `measure.py` | Encode what matters |
| Always exactly 10, all up front | `generate.py` | $3.00 committed before anything is seen |
| The inspection recipe | `SKILL.md` | Fixed, not driven by what the metrics flagged |

## Evidence from the two real runs

- **Run 1** (`20260813_214436`) reasoned to `cand_02, 10, 07, 05`, then typed the
  SKILL.md example `--picks 10,4,2,6` verbatim and shipped cand_04 (drift 22.7)
  and cand_06 (31.5), both REJECT rows. *Fixed:* `review.py` now refuses
  rejected picks, `measure.py` prints a `KEEP` line, the example is placeholders.
- **Run 1** burned 14k tokens (44% of context) reading `harness.py` and all five
  tool sources before doing anything, then compacted mid-inspection. *Fixed:*
  rules 1-2 in SKILL.md.
- **Run 2** (`20260813_220821`) was clean on all of the above - picks from KEEP,
  no source reading, 0 compactions, finished at 61% - but **every construction
  verdict was invalid**: it compared `(1700,1300)` on the source against
  `(950,700)` on a candidate, which is 28%/23% into the garment versus 1%/9%.
  Hip against waistband. *Fixed:* `crop_pair.py`.
- With matched boxes the same comparison returns a usable verdict: source shows
  vertical ribbed grain, `cand_10` shows uniform speckled noise - texture
  invented. The opposite of what run 2 concluded.

## What stays deterministic

Measurement. Masking, IoU, symmetry, tilt, centroid, colour distance, seam
energy and speck counting stay in scripts, because numbers have to be
reproducible across runs. `crop_pair.py` stays deterministic for the same reason
- it is geometry, not judgement.

---

## Change 1 - the agent writes the prompt

**Files:** `tools/prepare.py`, `tools/generate.py`, `task/SKILL.md`

`prepare.py` stops emitting a prompt. It confirms both inputs, reports
dimensions and mode, writes the downscaled upload copy, and writes
`archive/prompt_brief.md` - a checklist of the seven clauses a prompt must
cover, not a prompt. `PROMPT` and `SHAPES` are deleted.

The brief adapts to what was measured: if the reference is `mode=L` it tells the
agent the greyscale clause is mandatory and says why (the model reads grey as a
colour target and desaturates); if not, it says colour may bleed instead.

The agent then calls `compare_images` on off-set vs reference, and writes
`archive/prompt.txt` with a heredoc, describing the garment's actual edges and
naming the actual background clutter it saw.

`generate.py` refuses to run without `prompt.txt`, and warns if it is under ~120
words or missing any of `greyscale`/`flat`/`background`. Cheap guard against a
thin prompt costing $3.00.

**Risk:** a worse prompt than the hardcoded one. **Mitigation:** change 2 makes
that cost $0.90 instead of $3.00.

## Change 2 - probe at 2K in a loop, commit once at 4K

**Files:** `tools/generate.py`, `tools/measure.py`, `task/SKILL.md`

1K and 2K cost **$0.15**, 4K costs **$0.30**. So the prompt gets tested at half
price and 4K money is spent once, on a prompt that has already been proven.

```
loop (max 4 rounds):
    generate.py --run R --num 2 --resolution 2K --probe   $0.30/round
    measure.py  --run R --pattern 'probe_*.png'
    compare_images on both
    |- both fail the same way   -> rewrite prompt.txt, next round
    |- one fails, one is clean  -> seed luck, proceed
    '- both clean               -> proceed

final:
    generate.py --run R --num 6 --resolution 4K           $1.80
```

Typical 2-3 rounds plus the final = **$2.40-2.70**, under today's flat $3.00,
with a prompt that has been tested rather than assumed.

### Why 2 per probe and not 1

One sample cannot separate a bad prompt from a bad seed. Run 1's ten colour
drifts, all from the *same* prompt:

```
12.7  13.4  16.9  18.9  19.1 | 22.7  26.3  26.9  31.1  31.5
            kept             |        rejected
```

Seed variance spans 12.7 to 31.5. A single probe draws randomly from that - it
would condemn a working prompt on a 31, or bless a broken one on a 12. Two
samples give the only distinction that matters: **both failing the same way is a
prompt problem; one failing is noise.**

A single probe *is* sufficient for binary compliance - did the bench and shoes
go, did the hang tag go, is it flat, is it still green. Those fail in every
sample when the clause is missing. It is the graded metrics that need two.

### Why the final batch is 6, not 3

Both runs rejected exactly 5 of 10 on colour drift. At that rate 3 generations
yields 1-2 survivors and "best 4" is unreachable. 6 at 4K yields ~3 expected
survivors. If the probe loop improves the drift rate this can come down - which
is itself something the loop should reveal.

### Implementation notes

- `--probe` writes `archive/probe_<round>_<n>.png` instead of `cand_NN.png`, so
  cheap 2K probes never mix into the 4K set that gets measured and picked. Round
  number is inferred from what is already on disk.
- `--start N` on the final call, so a split 4K batch numbers continuously.
- `measure.py` gains `--pattern` (default `cand_*.png`) so probes can be measured
  with the same code.
- Probes stay in `archive/` as the record of what each prompt revision produced.
- Every round appends to `steps.log`, so total spend stays honest.

**Cost in context:** each round is one generate, one measure and two
`compare_images` - roughly 3-4k tokens. Four rounds would be ~14k, which on a
32k window means the loop needs a hard cap. Four rounds, then commit or stop.

## Change 3 - inspection follows the flags

**Files:** `task/SKILL.md`

Replace the fixed recipe with routing:

| Metric says | Look at |
|---|---|
| `seam/src` well above 1.0 | `--at waistband`, then `--at hem` - invented topstitching |
| `seam/src` well below 1.0 | `--at thigh` - cloudy paper instead of knit |
| `sym` below the reference's | `--at left` and `--at right`, compare to each other |
| `colour` near the reject bar | full-frame `compare_images` against the source |
| `spk` above 0 | full frame, name where the speck is |
| nothing flagged | `--at waistband` and `--at hem`, then stop |

Plus the rule already added: once two candidates agree on a defect, treat it as
common to the batch and move to a different region.

## Change 4 - thresholds stay as defaults, agent may override

**Files:** `task/SKILL.md` only

**This is a deliberate departure from "make it agentic".** Run 1 shows this
model will copy an example over its own reasoning; the reject rule is exactly
what caught it. Removing the guardrail invites a desaturated candidate straight
into `output/`.

So `measure.py` keeps `--reject-tilt` and `--reject-colour` and the score. The
agent may re-run with different values, and must state the number and the reason
in `## Notes`. Judgement with a floor under it.

Worth revisiting: colour drift alone rejected exactly 5 of 10 in **both** runs,
with survivors at 12.7-19.2 against a bar of 20.0. That bar is doing nearly all
the filtering and sits very close to the pack. It should be checked against
whether those five actually look desaturated before it is trusted further.

---

## Order of work

1. `prepare.py` - strip the prompt, write the brief (change 1)
2. `generate.py` - require and validate `prompt.txt`, add `--probe` and
   `--start` (changes 1, 2)
3. `measure.py` - add `--pattern` (change 2)
4. `task/SKILL.md` - rewrite steps 1, 2 and 4; add the routing table (all)
5. Dry-run the deterministic half on the existing run folder
6. One live run, then compare cost and picks against run 2

## Open question to settle on the first live run

Colour drift alone rejected exactly 5 of 10 in **both** runs, survivors at
12.7-19.2 against a bar of 20.0. That single threshold is doing nearly all the
filtering and sits very close to the pack. Before trusting it further, look at
two rejected candidates and confirm they are visibly desaturated. If they are
not, the bar is wrong and it has been discarding half of every batch.

## Not in scope

- Agent choosing the final batch size
- Agent editing the metrics themselves
- Repairing a bad candidate by feeding it back as an input. Worth one $0.30
  experiment separately, but it is close to the rule the log already sets -
  never re-roll a bad image with a fix prompt - and it is not part of this plan.
- Any change to `harness.py`, `review.py`, `crop_pair.py` or `common.py`
