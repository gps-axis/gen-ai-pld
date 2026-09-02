#!/usr/bin/env python3
"""The only billed step. One call per image, run concurrently.

    python tools/generate.py --run runs/<stamp> --num 2 --resolution 2K
    python tools/generate.py --run runs/<stamp> --num 1 --source cand_03
    python tools/generate.py --run runs/<stamp> --dry-run     # free, prints it

One image per call with its own seed, so each is an independent sample rather
than one batch the server draws together - which is what a shortlist needs.

`--source` is what makes iteration possible. It defaults to the segmented
original, but it takes any candidate name, so a generation can start from an
earlier attempt and fix one remaining defect instead of re-rolling the whole
thing. Every candidate's parent and chain depth are recorded in lineage.json,
because a chained edit drifts from the real garment in a way a single edit does
not, and depth is the only cheap warning of it.

Numbering continues from what is already in the folder, so topping up needs no
arguments. Candidates are numbered by SUBMISSION order, so cand_03 is always the
third seed however the wave happens to land.

`--max-total` is a hard ceiling on images per run, enforced here because task
text is advisory and has been overridden on real runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import metrics as M  # noqa: E402
import promptfile as P  # noqa: E402

# The operator's ceiling on images per run folder. Set it here, or per run with
# LAYDOWN_MAX_IMAGES=8 ./run.sh ...  --max-total can lower it but never raise it,
# because the agent chooses that flag and task text has been overridden on four
# separate runs.
HARD_CAP = int(os.environ.get("LAYDOWN_MAX_IMAGES", "10"))

# nano-banana-pro/edit takes a `system_prompt` alongside the scene prompt, and
# this project was not using it. That is the channel the duplicate-garment
# failure needed: the endpoint accepts a list of image_urls with no way to mark
# one of them as reference rather than content, so image 1 and image 2 arrive as
# two pictures that each contain a sweater and the model decides for itself what
# to do with the second one. Roughly two in ten came back holding both.
#
# Everything tried before this was scene-prompt persuasion, and the scene prompt
# is read as a description of the picture to make - which is why naming the
# failure there made it worse rather than better (8 in 10 on
# runs/20260827_110352). A system instruction is read as a standing rule about
# the job instead, so the role of each input can be stated once, up front, in
# the register where it belongs.
#
# Phrased positively throughout, for the reason recorded on the count clause
# below: naming the thing to avoid is how a diffusion model is told to draw it.
SYSTEM = (
    "You are a product photography retoucher. You produce catalogue flat lays: "
    "one garment, photographed from directly above, lying flat and square on a "
    "plain white background. "
    "The first input image is the product. Its construction, colour, pattern, "
    "texture and proportions are reproduced exactly as they are. "
    "Any further input image is a layout guide. It shows how the garment should "
    "be arranged and it is reference material rather than content: it "
    "contributes arrangement only, and its own subject stays out of the "
    "picture you produce. "
    "Every image you return is a single photograph of one garment."
)


# --------------------------------------------------------------------------
# Naming, lineage and the upload cache
# --------------------------------------------------------------------------

CAND_RE = re.compile(r"^cand_(\d+)$")          # generated - counts against budget
DERIVED_RE = re.compile(r"^cand_(\d+)[sp]+$")  # segmented / polished - not a buy


def generated(arch: Path) -> list[Path]:
    """Billed images only. cand_03s.png is a segmentation of cand_03, not a
    second purchase, so it must never move the counter or the numbering."""
    return sorted(p for p in arch.glob("cand_*.png") if CAND_RE.match(p.stem))


def resolve_source(run: Path, name: str) -> Path:
    """"source", a candidate name, or a path - to an actual file.

    Names rather than paths are the interface the model sees, so it cannot land
    a generation on something outside the run folder and orphan its lineage.
    """
    arch = Path(run) / "archive"
    if name in ("source", "src", "original", ""):
        return arch / "source_clean.jpg"
    if CAND_RE.match(name) or DERIVED_RE.match(name):
        return arch / f"{name}.png"
    return Path(name)


def lineage_path(run: Path) -> Path:
    return Path(run) / "archive" / "lineage.json"


def load_lineage(run: Path) -> dict:
    f = lineage_path(run)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def record(run: Path, name: str, entry: dict) -> None:
    data = load_lineage(run)
    data[name] = entry
    lineage_path(run).parent.mkdir(parents=True, exist_ok=True)
    lineage_path(run).write_text(json.dumps(data, indent=2) + "\n")


def depth_of(run: Path, source_name: str) -> int:
    """How many generations away from the original source this input already is.

    Segmentation does not count: it drops a background, it does not redraw
    anything, so a segmented candidate sits at its parent's depth. Only a fal
    call adds a hop, because only a fal call can invent detail.
    """
    if source_name in ("source", "src", "original", ""):
        return 0
    entry = load_lineage(run).get(source_name)
    if not entry:
        return 0
    return int(entry.get("depth", 0))


def uploads_path(run: Path) -> Path:
    return Path(run) / "archive" / "uploads.json"


def upload_cached(run: Path, path: Path, fal_client) -> str:
    """fal URL for `path`, keyed by CONTENT hash.

    Keyed by content rather than by "we already uploaded this process" because
    the same file is now sent across many separate generate calls in one run -
    the source goes up on every single one - and because a chained edit re-sends
    a candidate that was itself uploaded earlier. Hashing also means a file whose
    bytes changed uploads again instead of silently reusing a stale URL.
    """
    digest = C.md5(path)
    f = uploads_path(run)
    cache = {}
    if f.exists():
        try:
            cache = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    hit = cache.get(digest)
    if hit:
        print(f"  {path.name}: already uploaded (content hash {digest[:8]})")
        return hit
    print(f"  {path.name}: uploading...")
    try:
        url = fal_client.upload_file(str(path))
    except Exception as e:  # noqa: BLE001 - the model reads this, not a traceback
        detail = str(e)
        hint = ("FAL_KEY is not valid - check .env, or the environment the "
                "container was started with."
                if "401" in detail or "Unauthorized" in detail else
                "fal.ai could not be reached. Nothing was billed.")
        raise RuntimeError(f"upload of {path.name} failed: {hint}\n"
                           f"  ({type(e).__name__}: {detail[:200]})") from e
    cache[digest] = url
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cache, indent=2) + "\n")
    return url


# --------------------------------------------------------------------------
# The clause this script appends to every prompt
# --------------------------------------------------------------------------

def pose_clause() -> str:
    """Opt-in: take the LIMB GEOMETRY off image 2 instead of describing it.

    Off by default, and the reason is measured. Pointing at image 2 for pose is
    what produced 60-80% duplicate garments on runs/20260827_104532, _110352 and
    _111218, against 20-44% on the two runs before that clause existed. Told to
    reproduce image 2, the model reproduces image 2, and image 2 contains a
    garment. That is why every other pose instruction here is written in words.

    But words have a ceiling, and runs/20260827_222231 found it: the agent wrote
    "both sleeves splayed wide away from the body, angled down at roughly 30
    degrees, cuffs well clear of the body" - which is the reference, correctly
    described - and the generator returned the sleeves tucked in anyway. A sleeve
    angle is a geometric fact about a picture, and describing a picture in words
    is lossy in a way that pointing at it is not.

    So this exists, opt-in, for the case where the words have already failed.
    Three things keep the ghost risk down, and all three are deliberate:

      * it names the MEASUREMENT to copy - angle to the body, cuff distance from
        the sides, hem line - rather than "the pose", which reads as "the picture"
      * it is one sentence. The version that cost 60-80% was three emphatic ones
        ("image 2 is the authority ... follow image 2 every time")
      * it closes by re-anchoring the subject to image 1, so the last thing read
        is which garment is being drawn

    A duplicate that gets through is a bad draw, not a bad prompt: change the
    seed. That escape did not exist when this was first tried.
    """
    return (
        "\n\nFor the arrangement only, read the geometry off image 2: place the "
        "sleeves at the same angle away from the body as the sleeves in image 2, "
        "with the cuffs the same distance out from the sides and the hem sitting "
        "the same way. Geometry only - the garment being photographed is the one "
        "in image 1.\n")


def lay_clause(chained: bool) -> str:
    """The standing rules, appended to whatever the model wrote.

    Appended here rather than left to the model because every one of these was
    paid for by a measured failure, and because instructions in the skill have
    been skipped on four separate runs.
    """
    clause = ["\n\n---",
              # PHRASED POSITIVELY, ON PURPOSE. This clause used to spell the
              # failure out - "no second copy, no partial copy, no collar or
              # cuff of another garment, however faint" - and the duplicate rate
              # went UP, from 2 in 10 on runs/20260827_101946 to 8 in 10 on
              # runs/20260827_110352, the worst measured. Both clauses were
              # verified present in the sent prompt, so it was not ignored.
              #
              # Naming the thing eight times is how a diffusion model is told to
              # draw it. Negation is not reliably represented: the tokens that
              # land are "second copy", "another garment", "collar", and the
              # "no" in front of them carries far less weight than it does in
              # conversation. So the count is now stated once, as a fact about
              # what the frame contains, and the word for the failure never
              # appears.
              "The frame contains one garment, centred, surrounded by empty "
              "white background on all sides.",
              "",
              "IMAGE 2 IS A LAY REFERENCE ONLY. It shows how the garment should "
              "be ARRANGED - how it lies, how straps or legs are positioned, how "
              "square and symmetric the lay is. Take NOTHING else from it. Every "
              "seam, stitch line, neckline shape, panel, binding, trim and piece "
              "of hardware comes from image 1 and only from image 1. If image 2 "
              "shows construction image 1 does not have, it belongs to a "
              "different product: do not reproduce it.",
              "",
              # Also positive, for the same reason as the count above. This
              # named the poses to avoid - "creased, angled, arms folded or
              # turned in" - which is a list of things to draw as far as a
              # diffusion model is concerned. It describes the target pose only,
              # and settles the conflict by naming which image wins rather than
              # by describing what image 1 got wrong.
              #
              # DESCRIBED IN WORDS, not by pointing at image 2. The first
              # version said "MATCH THE POSE IN IMAGE 2 ... image 2 is the
              # authority ... follow image 2 every time", and the duplicate rate
              # tracked it exactly: every run carrying it came in at 60-80%
              # (runs/20260827_104532, _110352, _111218), against 20-44% on the
              # two runs before it existed (_101946, _095355). Told to match
              # image 2 and that image 2 wins, the model reproduces image 2 -
              # and image 2 contains a garment, so a second garment is drawn.
              #
              # TRIED AND REVERTED: a sentence describing the body outline -
              # "the body lies open and flat to its full width, each side seam
              # pulled straight from underarm to hem". Two runs carried it,
              # _123706 at 7/10 duplicates and _124623 at 6/10 against 2-6
              # before, and mean silhouette IoU went 0.849 without it, 0.813,
              # then 0.849 with it. Same outline, more ghosts. Every additional
              # sentence about the garment's own body is another mention of the
              # subject, and this model answers a repeated subject with a second
              # copy.
              # THE SLEEVE ANGLE USED TO BE HARDCODED HERE and it was wrong.
              # This sentence read "Sleeves or legs straight down at the sides,
              # evenly spaced and symmetric, each cuff clear of the body", which
              # was correct for exactly as long as the reference was auto-picked
              # from a 45-image house-style library where every laydown posed
              # the sleeves that way.
              #
              # The operator now supplies the reference, and it can pose the
              # sleeves any way it likes. On runs/20260827_215408 the reference
              # had them splayed wide at ~35 degrees; this clause said "straight
              # down at the sides", the agent wrote its own pose section to
              # AGREE with the clause - "both sleeves brought in from their
              # splayed position, close to the body" - and all eight candidates
              # came back with the sleeves tucked in. The clause did not just
              # lose to the reference, it talked the agent out of it.
              #
              # So this now carries only what is true of every laydown -
              # symmetry, level shoulders and hem, no tilt, centred, in frame -
              # and the sleeve and leg ANGLE belongs to the agent's own `pose`
              # section, written from looking at image 2.
              #
              # It stays described IN WORDS rather than as "match image 2":
              # pointing at image 2 for pose is what produced 60-80% duplicate
              # garments (runs/20260827_104532, _110352, _111218) against 20-44%
              # before it existed. Told to reproduce image 2, the model
              # reproduces image 2, and image 2 contains a garment.
              "LAY THE GARMENT SQUARE AND FLAT. Both sleeves arranged "
              "symmetrically with each other, each cuff clear of the body and "
              "clear of the hem. Shoulders level, hem level and parallel to the "
              "bottom of the frame, the garment upright and centred with no "
              "rotation or tilt, the whole garment inside the frame. This is how "
              "the finished flat lies, whatever arrangement the garment happens "
              "to be in when photographed.",
              "",
              # Keeps the scoping that fixed the folded-arm problem - "keep it
              # unchanged" was being read as covering pose - without pointing at
              # image 2 as something to reproduce.
              "Any instruction above to keep the garment unchanged, untouched or "
              "exactly as in image 1 refers to its CONSTRUCTION, COLOUR, PATTERN "
              "and TEXTURE only. It never refers to the pose. Re-laying the "
              "garment flat and square IS the task, so moving a sleeve, "
              "straightening a hem or squaring the body is required, not a "
              "change to be avoided.",
              "",
              # The flip is not a construction error and no construction clause
              # prevents it: every seam can be correct and the garment still
              # arrive inside-out or back-to-front, because the lay reference
              # shows a different face and the model reconciles the two.
              "Show the same face of the garment as image 1, exterior toward the "
              "camera - do not flip the garment or show its interior or reverse "
              "side. Image 2 defines only how flat and square the garment lies, "
              "never which side faces up.",
              "",
              # STANDING, on every prompt, because it has to be and because the
              # agent kept forgetting. Segmentation drops the BACKGROUND only -
              # its own docstring says so - so a pin, clip, tag or hanger holding
              # the garment up for the photograph survives into image 1 every
              # single time. Nothing else removes it. On runs/20260827_223611 not
              # one prompt section mentioned them and the pins duly shipped.
              #
              # The whole clause turns on ONE distinction, which is the thing the
              # agent got wrong in both directions: what was holding the garment
              # up is temporary and goes; what is sewn into the garment is the
              # product and stays. Stated as a description of the finished
              # photograph rather than as a list of things not to draw, because
              # negation is what made the old tag failure worse.
              #
              # Asking for removal DOES work here, which is worth recording
              # because the old skill implied the opposite: a run whose prompt
              # named "the small pin or hook and thread loop at the top centre of
              # the fur collar" came back with it gone. The failure that scared
              # everyone off was a prompt asking a tag to STAY IN PLACE, which is
              # the opposite instruction.
              "The finished photograph shows the garment by itself. Any pin, "
              "clip, tack, hanger, hook, price ticket or swing tag that was "
              "holding it in place for the shot is gone, and the fabric it was "
              "holding lies flat and closed where it used to be.",
              "",
              "Everything sewn into the garment stays exactly as it is and is "
              "reproduced unchanged: every seam and stitch line, the sewn-in "
              "brand and care labels, embroidery, appliqué, and any printed, "
              "woven or knitted logo. Those are the product."]

    if chained:
        # NEW WITH CHAINING, AND UNMEASURED - there is no run history behind
        # this one yet, unlike everything above it.
        #
        # The reasoning: on a chained edit, image 1 is no longer a photograph.
        # It is an earlier generation, so its construction has already been
        # through the model once and any invented detail in it now arrives
        # labelled "the product, reproduced exactly as it is". Left unsaid, the
        # standing rules above would ask for that invention to be preserved
        # faithfully, and each further hop would harden it. Saying image 1 is an
        # in-progress retouch of a real garment is the closest available framing
        # to the truth.
        clause += ["",
                   "Image 1 is an earlier retouch of this product, not the "
                   "original photograph. Treat it as work in progress: keep what "
                   "is already correct, change only what the instructions above "
                   "ask for, and where a detail looks uncertain or invented, "
                   "render it plainly rather than elaborating on it."]

    return "\n".join(clause) + "\n"


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="default <run>/archive/reference_greyscale.jpg")
    ap.add_argument("--source", default="source",
                    help="'source', a candidate name like cand_03, or a path")
    ap.add_argument("-n", "--num", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=5,
                    help="calls in flight. Lower if fal rate-limits.")
    ap.add_argument("--base-seed", type=int, default=1000,
                    help="image i uses base_seed + i, recorded in seeds.json")
    ap.add_argument("--seed", type=int, help="force one seed for every image")
    ap.add_argument("--resolution", default="2K", choices=("1K", "2K", "4K"))
    ap.add_argument("--aspect-ratio", default=C.ASPECT)
    ap.add_argument("--max-total", type=int, default=HARD_CAP,
                    help=f"images per run folder. Ceiling {HARD_CAP}; "
                         f"this can lower it, never raise it.")
    ap.add_argument("--match-pose", action="store_true",
                    help="Take the sleeve angle and cuff spacing off image 2 "
                         "instead of from the prompt's words. Use when the words "
                         "have already failed. Raises the duplicate-garment rate "
                         "- change the seed if one appears.")
    ap.add_argument("--no-reference", action="store_true",
                    help="send image 1 only, no lay reference")
    ap.add_argument("--force", action="store_true",
                    help="send a prompt the guardrails refused, on the record")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the assembled prompt and bill nothing")
    a = ap.parse_args()

    run = a.run
    arch = run / "archive"
    arch.mkdir(parents=True, exist_ok=True)
    reference = a.reference or C.reference_path(run)

    src = resolve_source(run, a.source)
    if not src.exists():
        print(f"Source not found: {src}")
        if a.source != "source":
            have = [p.stem for p in sorted(arch.glob("cand_*.png"))]
            print(f"  candidates on disk: {', '.join(have) or '(none)'}")
        return 1

    use_ref = not a.no_reference
    if use_ref and not reference.exists():
        print(f"Reference not found: {reference}  (--no-reference sends image 1 "
              f"alone)")
        return 1

    # ---- the prompt, and the guardrails, before any money --------------
    prompt = P.render(run)
    if not prompt.strip():
        print(f"The prompt is empty. Fill it in with prompt_set, or "
              f"tools/promptfile.py --run {run} --show to see the slots.")
        return 1

    problems, warnings = C.check_prompt(prompt, reference if use_ref else None)
    for w in warnings:
        print(f"prompt WARNING: {w}")
    if problems and not a.force:
        print(f"REFUSING TO GENERATE: the prompt is not ready.")
        for p in problems:
            print(f"  - {p}")
        print("  Fix the section that carries it, or force=true to send it as "
              "it stands.")
        C.log(run, f"generation REFUSED: prompt ({len(problems)} problem(s), "
                   f"{problems[0][:40]})")
        return 1
    if problems:
        print(f"--force: sending a prompt with {len(problems)} unresolved "
              f"problem(s)")
        for p in problems:
            print(f"  - {p}")

    parent_depth = depth_of(run, a.source)
    chained = a.source not in ("source", "src", "original", "")
    prompt = prompt.rstrip() + lay_clause(chained)
    if a.match_pose:
        if not use_ref:
            print("--match-pose needs image 2; it does nothing with "
                  "--no-reference.")
        else:
            prompt = prompt.rstrip() + pose_clause()
            print("--match-pose: the sleeve angle and cuff spacing come off "
                  "image 2 rather than from your words.")
            print("  This raises the odds of a SECOND garment in the frame - it "
                  "ran 60-80% on the three runs that last pointed at image 2 for "
                  "pose. Buy one or two, look, and change the seed if a "
                  "duplicate turns up; the prompt is not what is wrong when it "
                  "does.")

    if chained:
        print(f"chained edit: image 1 is {a.source} (depth {parent_depth}); the "
              f"result will be depth {parent_depth + 1}")
        if parent_depth >= 2:
            print("  NOTE: two edits deep is where invented detail starts to "
                  "harden. Measure against the ORIGINAL source, not against "
                  f"{a.source}.")
    print("prompt carries the layout-only clause for image 2"
          if use_ref else "no lay reference: image 1 only")

    phash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    snap = arch / f"prompt_{phash}.txt"
    if not snap.exists():
        snap.write_text(f"<!-- system: {SYSTEM} -->\n\n{prompt}")

    if a.dry_run:
        print("\n" + "=" * 70)
        print(prompt)
        print("=" * 70)
        print(f"\n--dry-run: nothing uploaded, nothing billed. "
              f"Snapshot at {snap}")
        return 0

    # ---- the budget -----------------------------------------------------
    cap = min(a.max_total, HARD_CAP)
    have = generated(arch)
    left = cap - len(have)
    if left <= 0:
        print(f"REFUSING TO GENERATE: the image budget is spent - "
              f"{len(have)}/{cap} used.")
        print("  Everything on disk is already paid for. Pick the best of what "
              "you have and finish.")
        C.log(run, f"generation REFUSED: budget spent ({len(have)}/{cap})")
        return 1
    want = min(a.num, left)
    if want < a.num:
        print(f"budget: asked for {a.num}, {left} left of {cap} - generating "
              f"{want}.")

    C.fix_ca_bundle()
    C.load_fal_key()
    import fal_client
    import requests

    try:
        urls = [upload_cached(run, src, fal_client)]
        if use_ref:
            urls.append(upload_cached(run, reference, fal_client))
    except RuntimeError as e:
        print(f"REFUSING TO GENERATE: {e}")
        C.log(run, "generation REFUSED: upload failed")
        return 1

    start = len(have)
    # --seed sets the base for the WAVE, not one seed for every image in it.
    # It used to be the latter, so `--seed 42 --num 2` sent seed 42 twice and
    # bought the same picture at full price. A seed is the knob for escaping a
    # bad draw, and the commonest way to use it is "try somewhere else, a couple
    # at a time" - which was exactly the call that silently paid double.
    base = a.seed if a.seed is not None else a.base_seed + start
    seeds = [base + i for i in range(want)]
    names = [f"cand_{start + i + 1:02d}" for i in range(want)]

    def one(i: int):
        args = {"prompt": prompt, "image_urls": urls,
                "num_images": 1, "output_format": "png",
                "system_prompt": SYSTEM,
                "resolution": a.resolution, "aspect_ratio": a.aspect_ratio,
                "seed": seeds[i]}
        for attempt in (1, 2):
            try:
                r = fal_client.subscribe(C.ENDPOINT, arguments=args,
                                         with_logs=False)
                items = r.get("images", [])
                if not items:
                    raise RuntimeError("no images returned")
                img = Image.open(requests.get(items[0]["url"], stream=True,
                                              timeout=300).raw).convert("RGB")
                return i, img, None
            except Exception as e:  # noqa: BLE001 - one bad call must not stop the wave
                if attempt == 2:
                    return i, None, str(e)
                time.sleep(3)
        return i, None, "unreachable"

    print(f"generating {want} at {a.resolution} from {a.source}, prompt {phash}")
    got, failed = [], []
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = [pool.submit(one, i) for i in range(want)]
        for fut in as_completed(futures):
            i, img, err = fut.result()
            if err:
                failed.append((names[i], err))
                print(f"  {names[i]}: FAILED - {err[:120]}")
                continue
            out = arch / f"{names[i]}.png"
            img.save(out)
            record(run, names[i], {
                "parent": a.source,
                "parent_kind": "generation" if chained else "source",
                "depth": parent_depth + 1,
                "prompt_hash": phash,
                "seed": seeds[i],
                "resolution": a.resolution,
                "reference": str(reference) if use_ref else None,
                "created": datetime.now().isoformat(timespec="seconds"),
            })
            got.append(names[i])
            print(f"  {names[i]}: saved")

    if not got:
        C.log(run, f"generation produced nothing ({len(failed)} failed)")
        return 1

    # seeds.json, merged - one record per image, never overwritten wholesale
    sf = arch / "seeds.json"
    book = {}
    if sf.exists():
        try:
            book = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            book = {}
    for i, name in enumerate(names):
        if name in got:
            book[name] = {"seed": seeds[i], "resolution": a.resolution,
                          "prompt": phash, "source": a.source}
    sf.write_text(json.dumps(book, indent=2) + "\n")

    cents = len(got) * (C.PRICE_4K if a.resolution == "4K" else 0.15) * 100
    total = len(generated(arch))
    C.log(run, f"generated {len(got)} at {a.resolution} from {a.source}, "
               f"prompt {phash} ({total} total)", cents)

    # ---- the free reading, attached to every new candidate --------------
    # Run here rather than left to the model to ask for: the redraw is the one
    # failure eyes miss, and a number nobody requested is a number that arrives
    # in time to change the next decision.
    source_clean = arch / "source_clean.jpg"
    rows = []
    if source_clean.exists():
        print()
        for name in got:
            try:
                m = M.compare(source_clean, arch / f"{name}.png")
                m["depth"] = parent_depth + 1
                rows.append(m)
                print("  " + M.line(m, name))
            except Exception as e:  # noqa: BLE001 - a measurement is not the delivery
                print(f"  {name}: could not measure ({type(e).__name__}: {e})")
    else:
        print(f"\n  (no {source_clean.name}, so nothing to measure against)")

    (arch / "last_generation.json").write_text(json.dumps({
        "candidates": got, "failed": [n for n, _ in failed],
        "source": a.source, "depth": parent_depth + 1,
        "prompt_hash": phash, "resolution": a.resolution,
        "cents": cents, "used": total, "cap": cap,
        "metrics": rows,
    }, indent=2) + "\n")

    print(f"\n{len(got)} image(s), {cents:.1f}c. Budget {total}/{cap} used, "
          f"{cap - total} left.")
    if failed:
        print(f"{len(failed)} failed and cost nothing: "
              f"{', '.join(n for n, _ in failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
