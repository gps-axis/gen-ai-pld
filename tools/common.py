"""Shared helpers for the laydown tools.

Everything here is deterministic and cheap. The billed work is in generate.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
RUNS = ROOT / "runs"

# nano-banana-pro/edit at 4K is $0.30/image; 1K and 2K are both $0.15. Published
# rates - fal exposes no billing API, so a reported cost is never a receipt.
ENDPOINT = "fal-ai/nano-banana-pro/edit"
PRICE_4K = 0.30

# The shape every generation comes back in. Deliberately fixed rather than
# derived from the input: the deliverable is a 3:4 flat whatever the photographer
# handed over, and matching the output to an odd source would just move the odd
# shape downstream. generate.py's --aspect-ratio still overrides it.
ASPECT = "3:4"


def pad_to_aspect(im, aspect: str = ASPECT, fill=(255, 255, 255)):
    """Letterbox an image to `aspect` on a white plate, garment centred.

    This is what stops the ghost. The source photos are landscape (4096x3072)
    while the output is portrait 3:4, so the model was handed a wide picture and
    asked for a tall one. Laying the garment square in that taller frame leaves
    an empty band above it, and a band of nothing is where an image model puts a
    second copy of the subject: runs/20260827_095355 came back with four of nine
    candidates showing a whole extra sweater or the top of one fading in, and
    runs/20260827_101946 still had two of ten after everything else was fixed.

    Padding the SOURCE to the target shape removes the discrepancy at the input
    rather than the output. The model then sees a 3:4 frame containing a garment
    with white margins and has nothing left to invent. 3:4 out is unchanged,
    which is the point - the odd shape is absorbed here instead of being passed
    on.

    A source already at the target ratio is returned untouched.
    """
    w, h = im.size
    aw, ah = (float(x) for x in aspect.split(":"))
    want = aw / ah
    have = w / h
    if abs(have - want) < 0.01:
        return im
    if have > want:                      # too wide - grow the height
        nw, nh = w, int(round(w / want))
    else:                                # too tall - grow the width
        nw, nh = int(round(h * want)), h
    out = Image.new("RGB", (nw, nh), fill)
    out.paste(im.convert("RGB"), ((nw - w) // 2, (nh - h) // 2))
    return out

# Canonical region profile -> the garment_type spellings that mean it. This is
# the same token vocabulary match_reference.py uses to pick the library folder,
# so the crops a tool measures with always follow the category step 0 already
# decided rather than a flag somebody remembered to pass. Shorts are graded on
# the leggings bands: same waistband, same hip, hem just higher up.
PROFILE_TERMS = {
    "bras": {"bra", "bras", "sports_bra", "sports_bras", "bralette",
             "bralettes", "t_shirt_bra", "crop_top", "crop_tops"},
    "leggings": {"legging", "leggings", "legging_bottoms", "tight", "tights",
                 "short", "shorts", "biker_short", "biker_shorts",
                 "cycling_shorts"},
    # Full-length woven bottoms. Split from leggings in match_reference for the
    # LAY, which differs a lot, but the crop geometry barely differs at all and
    # that is a measurement rather than an assumption: drawn on the upload from
    # runs/20260827_131825, the leggings bands land on 0.00-0.22 waistband, belt
    # loops, button and the top of the fly; 0.18-0.45 fly, rivets, pocket bags
    # and the rise; 0.80-1.00 both leg hems. So the bands are inherited and only
    # renamed, in REGIONS_BY_PROFILE, for what a woven bottom actually has.
    #
    # Terms are disjoint from leggings - profile_for() returns the first set a
    # token appears in, so a word in both would resolve by dict order.
    "loose": {"loose", "loose_fit", "jean", "jeans", "denim", "denims",
              "pant", "pants", "trouser", "trousers", "jogger", "joggers",
              "sweatpant", "sweatpants", "chino", "chinos", "cargo",
              "cargo_pant", "cargo_pants", "wide_leg", "barrel_leg"},
    # Gathered-waist woven bottoms. A separate library category from loose,
    # because the waistband changes the lay - but the same three crop bands,
    # checked the same way: drawn on three boyfriend/ assets with a paperbag, a
    # smocked and an elasticated band, 0.00-0.22 holds the frill, the elastic
    # and the closure below it, 0.18-0.45 holds the pockets and the rise, and
    # 0.80-1.00 holds both hems. Only the first band's NAME changes, because a
    # frill standing off the plate is the thing most likely to be generated
    # wrong here and "waistband/fly" does not ask about it.
    #
    # Terms are disjoint from loose and leggings - profile_for() returns the
    # first set a token appears in, so a word in both would resolve by dict
    # order.
    "boyfriend": {"boyfriend", "boyfriends", "boyfriend_fit", "boyfriend_jean",
                  "boyfriend_jeans", "boyfriend_pant", "boyfriend_pants",
                  "paperbag", "paperbag_pant", "paperbag_pants", "paper_bag",
                  "slouch", "slouchy", "slouch_fit"},
    # Sleeved upper body. It shares the bra's orientation - collar at the top,
    # hem at the bottom - and nothing else: the sleeves push the bounding box
    # out sideways, so a bra's 'cups' band lands half on sleeve and half on
    # plate. Its own bands, in both region tables.
    "pullovers": {"pullover", "pullovers", "sweater", "sweaters", "sweatshirt",
                  "sweatshirts", "jumper", "jumpers", "hoodie", "hoodies",
                  "hooded_sweatshirt", "crewneck", "crewnecks", "crew_neck",
                  "half_zip", "quarter_zip", "long_sleeve_top"},
    # Front-opening sleeved knit. Same three bands as pullovers, checked the
    # same way: drawn on a striped, a shawl-collar and a cropped cardigan,
    # 0.00-0.28 holds the neckline and the top of the placket, 0.22-0.62 the
    # placket, the middle buttons and the sleeves, 0.76-1.00 the hem rib, the
    # bottom button and the cuffs. All three carry the placket, which is the
    # point - it is the construction a generation gets wrong, and it runs the
    # full height of the garment rather than living in one band.
    #
    # Terms are disjoint from pullovers and fleeces - profile_for() returns the
    # first set a token appears in, so a word in both would resolve by dict
    # order.
    "cardigans": {"cardigan", "cardigans", "cardi", "cardis", "button_front",
                  "button_up", "button_through", "open_front", "duster",
                  "shrug", "shrugs", "bolero", "boleros"},
    # Also sleeved upper body, and deliberately NOT folded into pullovers. The
    # silhouette is close, but the pile sits proud of the seam: loft pushes the
    # body out ~2% of the bbox each side, so the pullover's inset boxes clip the
    # edge of a fleece, and the construction that has to survive generation is
    # different - a zip placket running the full length, hand pockets at hip
    # height, a stand collar or hood taller than any crewneck. Those get their
    # own bands rather than being judged inside a box drawn for a sweatshirt.
    #
    # Terms are kept disjoint from pullovers on purpose: profile_for() returns
    # the first set a token appears in, so a word in both would resolve by dict
    # order. A hoodie is a pullover here unless the garment_type says fleece.
    "fleeces": {"fleece", "fleeces", "fleece_top", "fleece_jacket",
                "fleece_pullover", "polar_fleece", "microfleece",
                "micro_fleece", "sherpa", "sherpas", "sherpa_jacket",
                "sherpa_pullover"},
}
DEFAULT_PROFILE = "leggings"


def session_run_dir() -> Path:
    """The one run folder for this session.

    run.sh stamps LAYDOWN_SESSION once and exports it, so every call resolves to
    the same folder. That is what makes the image budget hold: the cap counts
    images in a folder, so a second folder would be a second budget, and the
    agent creates folders by calling prepare.py.
    """
    sid = os.environ.get("LAYDOWN_SESSION")
    return RUNS / (sid or time.strftime("%Y%m%d_%H%M%S"))


def fix_ca_bundle() -> None:
    """certifi's Mozilla-only bundle rejects the corporate proxy's certificate;
    the Homebrew OpenSSL store includes it."""
    bundle = "/opt/homebrew/etc/openssl@3/cert.pem"
    if os.path.exists(bundle):
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)


def conf(key: str, default: str = "") -> str:
    """A setting from the environment, else the project .env, else the default.

    The harness passes its own .env down to every tool it spawns, so under a
    normal run the environment already has these. The .env fallback is for a
    tool run by hand from a shell that exported nothing - which is how they get
    debugged, and which used to mean silently falling back to a stale default.
    """
    v = os.environ.get(key)
    if v:
        return v
    env = ROOT / ".env"
    if env.exists():
        m = re.search(rf'^\s*{re.escape(key)}\s*=\s*["\']?([^"\'\s]+)',
                      env.read_text(), re.M)
        if m:
            os.environ[key] = m.group(1)
            return m.group(1)
    return default


def load_fal_key() -> str:
    key = conf("FAL_KEY")
    if key:
        return key
    sys.exit("FAL_KEY not set and none found in .env")


def log(run_dir: Path, what: str, cents: float = 0.0) -> str:
    """Append one line to <run>/steps.log and echo it.

    The running total is read back off the last line, so separate processes keep
    one continuous tally per run folder.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "steps.log"
    n, total = 0, 0.0
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.search(r"total\s+([0-9.]+)c\s*$", line)
            if m:
                n, total = n + 1, float(m.group(1))
    total += cents
    line = (f"[{n + 1}] {time.strftime('%H:%M:%S')}  {what:<46} "
            f"{cents:6.1f}c   total {total:.1f}c")
    with p.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return line


def md5(p: Path) -> str:
    """Content fingerprint of a file. Used to tell "the same image" from "an
    image with the same name", which is the whole basis of reusing a verdict."""
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def profile_for(garment_type: str) -> str | None:
    """Canonical profile for a garment_type string, or None if it means nothing
    here. Matching is on the normalised token, so 'sports_bra', 'Sports Bra'
    and 'bras' all land on the same profile."""
    w = re.sub(r"[^a-z]+", "_", str(garment_type).lower()).strip("_")
    for prof, terms in PROFILE_TERMS.items():
        if w in terms:
            return prof
    return None


def garment_profile(run_dir: Path,
                    override: str | None = None) -> tuple[str, str, bool]:
    """Which region profile this run's garment needs: (profile, why, resolved).

    Step 0 already classified the garment - select_reference.py writes the
    garment_type it used to pick the library folder into
    <run>/reference_selection.json - so no tool needs to be told again, and a
    flag left off cannot quietly point a region-based check at the wrong part of
    the garment. That failure is silent and expensive: a bra measured with the
    leggings bands puts 'waistband' on empty plate above the straps and 'hem'
    below the garment entirely, and every candidate then comes back flagged in
    near-identical words because the crops, not the candidates, are wrong.

    `resolved` is False when the garment could not be established, and the
    profile returned alongside it is only a fallback. `why` says what was found
    and no more - what a tool then does about it differs by tool, so each one
    states its own consequence rather than inheriting a sentence from here.
    Callers print `why` either way: a profile that was guessed has to be visible
    in the output it produced.
    """
    if override:
        prof = profile_for(override) or override
        return prof, f"{prof} - forced by --profile", prof in PROFILE_TERMS

    f = Path(run_dir) / "reference_selection.json"
    if not f.exists():
        return (DEFAULT_PROFILE,
                f"WARNING: no {f.name} in {Path(run_dir).name}, so the garment "
                f"is unknown", False)
    try:
        sel = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return (DEFAULT_PROFILE,
                f"WARNING: {f.name} unreadable ({type(e).__name__}), so the "
                f"garment is unknown", False)

    gt = str((sel.get("query_attrs") or {}).get("garment_type", "")).strip()
    prof = profile_for(gt)
    if prof:
        return prof, f"{prof} - from garment_type '{gt}' in {f.name}", True
    return (DEFAULT_PROFILE,
            f"WARNING: garment_type {gt or 'missing'!r} in {f.name} matches no "
            f"profile ({', '.join(sorted(PROFILE_TERMS))})", False)


# How far below the plate a pixel has to sit before it counts as garment.
# Named because two things now depend on agreeing exactly where the garment
# ends: garment_mask, and clean.py's guard on what the generative eraser is
# allowed to change. If they disagreed the guard would police a different
# outline from the one every downstream measurement uses.
PLATE_MARGIN = 18.0

# ...but 18 is a FLOOR, not the rule. It hugs the plate, which is right for a
# pale garment a few levels below it and wrong for a dark one, because
# everything between the two - including the garment's own contact shadow - then
# counts as garment.
#
# Measured on the black-and-cream bra: plate 209, garment core 52. At `plate -
# 18` the mask's bottom rows run 161, 186, 188 - shadow, not fabric, sitting
# nearly at plate level. They put the source outline 16 rows below the real hem.
# Matting then removes the shadow, as it should, and the outline gate reads the
# difference as the garment losing its bottom edge: "bottom edge pulled in 3.0%
# (limit 2%)", "outline lost 10.4% (limit 10%)". Both hairline, both false, and
# both fatal to the run.
#
# So the threshold sits a fraction of the way from the plate towards the
# garment's own tone. A pale garment (contrast ~25) keeps the 18 floor; a black
# one on a light bench (contrast 158) gets 55, which is below any penumbra and
# far above the fabric.
SHADOW_FRAC = 0.35

# The same question asked in colour instead of brightness: how far a pixel's own
# colour cast has to sit from the plate's before it counts as garment, in 0-255
# channel units.
#
# Luminance alone is not enough and this project has the bill for it. The mint
# bralette in runs/20260819_205617 sits 25.6 luminance below its plate on the
# cleaned upload and less than that on several candidates, so `a < plate - 18`
# either takes the whole garment or takes a shadow instead, with nothing in
# between. Measured on that batch, the luminance mask collapsed to a 3.2-3.7%
# strip of frame on cand_02, cand_03 and cand_04 - a band close-up, aspect 2.1
# to 2.6 - while the mask on cand_01 and cand_05 covered a plausible 21-26%.
# Every crop, every silhouette and every contact sheet built from that mask was
# then measuring a different thing per candidate.
#
# Chroma separates the same batch cleanly: garment-to-plate chroma runs 8.1-14.0
# on all ten, and the plate's own border noise runs 1.7-3.0. So the threshold is
# a floor plus that measured noise, not a constant.
CHROMA_MARGIN = 4.0
CHROMA_NOISE_PAD = 2.0
# Below these the chroma cue is not trusted and luminance is used instead: a
# white, grey or black garment on a neutral plate has no chroma to find, and
# thresholding noise would return a blob of nothing.
#
# CHROMA_SEPARATION is how far above its own threshold the chroma inside the
# blob has to sit before the blob is believed to be a coloured object. Across
# the twelve images of runs/20260819_205617 - the raw phone photo, the cleaned
# upload and ten candidates - that ratio measured 1.57 to 2.89, the 1.57 being
# the raw photo where the room light drags the plate off neutral. A blob
# thresholded out of noise sits near 1.0, so 1.4 has margin on both sides.
CHROMA_MIN_AREA = 0.005
CHROMA_SEPARATION = 1.4

# Clearing that bar is NOT enough to be used. Both cues are measured and the
# one that finds MORE of the garment wins, because both ways of failing look
# the same: the losing cue comes back with a fragment.
#
# runs/20260819_205617, pale mint on white: luminance fell to a 3.2-3.7% shadow
# strip on three candidates while chroma held 24-32%.
# runs/20260820_115631, near-black on white: chroma fell to a 5.9% beige strip
# at the right edge - a strip 0.34 as wide as it is tall - while luminance held
# 31.6%. Its chroma separation was 4.3x, comfortably over the bar above, and it
# was still the wrong object. Size is what separated the two cases, both times.
#
# A mask covering more than PLAUSIBLE_MAX of the frame is leaking into the
# background rather than segmenting, so it does not get to win on size.
#
# Chroma has to clear CHROMA_SEPARATION as well, so a blob thresholded out of
# noise cannot win by being large. Luminance has no equivalent test and does not
# need one here: every source in this pipeline is a garment on a light bench or
# a light plate, so a big dark region is the garment.
#
# The two cues are NOT forced to agree across images by this rule - they cannot
# be, since it is decided per image. Anything comparing two images passes `cue`
# explicitly instead.
PLAUSIBLE_MAX = 0.70


def _gray(path: Path, long_side: int = 1024) -> np.ndarray:
    im = Image.open(path).convert("L")
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def _rgb(path: Path, long_side: int = 1024) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def _border_px(a: np.ndarray, border: int = 8) -> np.ndarray:
    """Every pixel in the outer frame, as a flat list. Colour images keep their
    channels, so a plate colour can be taken the same way a plate level is."""
    b = border
    edges = [a[:b], a[-b:], a[:, :b], a[:, -b:]]
    if a.ndim == 3:
        return np.concatenate([e.reshape(-1, a.shape[-1]) for e in edges])
    return np.concatenate([e.ravel() for e in edges])


def _plate_from(a: np.ndarray, border: int = 8) -> float:
    return float(np.median(_border_px(a, border)))


def plate_level(path: Path, long_side: int = 1024) -> float:
    """Luminance of the backdrop, read off the image's own border.

    The plate is NOT white - it sweeps 228-252 on this project - so anything
    asking "is this pixel background?" has to measure the level rather than
    assume 255.
    """
    return _plate_from(_gray(path, long_side))


def _chroma(a: np.ndarray, plate_rgb: np.ndarray) -> np.ndarray:
    """How far each pixel's colour cast is from the plate's, per channel.

    The cast is the pixel minus its own mean, so it carries no brightness at
    all: a mint garment reads the same whether it sits in light or in shadow,
    which is exactly why a shadow on a neutral plate scores ~0 here and a pale
    garment does not.
    """
    cast = a - a.mean(axis=2, keepdims=True)
    plate_cast = plate_rgb - plate_rgb.mean()
    return np.abs(cast - plate_cast).max(axis=2)


def _blob(mask: np.ndarray, fill: bool = True) -> np.ndarray:
    """Open, fill and keep the largest connected component.

    Without the largest-blob step a single speck of dust moves the bbox by 80%
    while every aggregate number stays plausible.

    `fill=False` keeps the garment's own holes - the gap between a bra's
    crossed straps, the space between two legs. Every measurement here wants
    them filled; a line drawing of the garment does not, because those holes
    are most of what makes the lay readable.
    """
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))
    if fill:
        mask = ndimage.binary_fill_holes(mask)
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == int(np.argmax(sizes)) + 1


def _core(mask: np.ndarray, k: int = 9) -> np.ndarray:
    """Well inside the mask, so the soft plate-to-garment edge never speaks for
    what the garment is."""
    inner = ndimage.binary_erosion(mask, np.ones((k, k)))
    return inner if inner.sum() > 200 else mask


def garment_evidence(path: Path, long_side: int = 1024,
                     cue: str | None = None) -> dict:
    """Where the garment is, and which cue found it.

    Both cues are measured on every image:

      * **chroma** - the pixel's colour cast against the plate's own. It
        ignores shadows and survives a garment only a few levels darker than
        its plate, which is where luminance gives up.
      * **luminance** - the original rule, `a < plate - PLATE_MARGIN`. Right
        for a white, grey or black garment, where there is no chroma to read.

    Whichever finds more of the garment wins - see the constants above for the
    two runs that decided that rule, one in each direction.

    `cue` forces one of them. Pass it whenever two images are about to be
    compared: a source measured on luminance against a cleaned copy measured on
    chroma is two different objects, and the difference between them gets
    reported as damage to the garment. That is not hypothetical - it stopped
    runs/20260820_115631 before it started, with "outline lost 81.3% of its
    area" on a clean that had not lost anything.

    The dict carries the numbers the decision was made on, so a tool can print
    why it looked where it did rather than asserting it looked in the right
    place. Downscaled for speed: every metric taken from this is a shape or
    colour statistic and none of them changes under a clean 1024px resample.
    """
    a = _rgb(path, long_side)
    L = a.mean(axis=2)
    plate_rgb = np.median(_border_px(a), axis=0)
    plate_L = float(plate_rgb.mean())

    chroma = _chroma(a, plate_rgb)
    noise = float(np.percentile(_border_px(chroma), 99))
    thr = max(CHROMA_MARGIN, noise + CHROMA_NOISE_PAD)

    chroma_mask = _blob(chroma > thr)
    # Two passes: the floor finds the garment, the garment's own tone then says
    # where the fabric stops and its shadow starts. See SHADOW_FRAC.
    luma_margin = PLATE_MARGIN
    luma_mask = _blob(L < plate_L - luma_margin)
    if luma_mask.any():
        contrast_0 = plate_L - float(L[_core(luma_mask)].mean())
        luma_margin = max(PLATE_MARGIN, SHADOW_FRAC * contrast_0)
        if luma_margin > PLATE_MARGIN:
            luma_mask = _blob(L < plate_L - luma_margin)
    inside = float(chroma[_core(chroma_mask)].mean()) if chroma_mask.any() else 0.0

    def plausible(m: np.ndarray) -> float:
        """Area, or 0 for a mask that is too small to be a garment or so large
        it has swallowed the backdrop."""
        f = float(m.mean())
        return f if CHROMA_MIN_AREA <= f <= PLAUSIBLE_MAX else 0.0

    chroma_ok = (plausible(chroma_mask) > 0 and inside >= CHROMA_SEPARATION * thr)
    if cue in ("chroma", "luminance"):
        use_chroma, why = cue == "chroma", f"forced to {cue}"
    elif chroma_ok and plausible(chroma_mask) > plausible(luma_mask):
        use_chroma, why = True, "chroma finds more of the garment"
    elif plausible(luma_mask) > 0:
        use_chroma, why = False, "luminance finds at least as much"
    elif chroma_ok:
        use_chroma, why = True, "luminance found nothing usable"
    else:
        use_chroma, why = False, "neither cue is convincing"

    cue_raw = (chroma > thr) if use_chroma else (L < plate_L - luma_margin)
    mask = chroma_mask if use_chroma else luma_mask
    contrast = plate_L - float(L[_core(mask)].mean()) if mask.any() else 0.0

    ys, xs = np.nonzero(mask)
    box = ([int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
           if len(xs) else None)
    return {
        "mask": mask,
        # The same blob with its holes left open. Only outline_map() wants this.
        "mask_open": _blob(cue_raw, fill=False),
        # Every cue pixel, BEFORE _blob() throws away all but the largest
        # component. extra_garments() needs this and nothing else does: the
        # largest-blob step is exactly what made a second sweater invisible to
        # every metric in the pipeline.
        "cue_raw": cue_raw,
        "shape": a.shape[:2],
        "cue": "chroma" if use_chroma else "luminance",
        "cue_why": why,
        # What the cue that LOST would have found. A tool comparing two images
        # can check that both landed on the same cue, and say so when they did
        # not - the alternative is reporting the disagreement as garment damage.
        "area_chroma": round(float(chroma_mask.mean()), 4),
        "area_luminance": round(float(luma_mask.mean()), 4),
        "plate_rgb": plate_rgb,
        "plate_level": plate_L,
        "chroma_threshold": round(thr, 2),
        "chroma_inside": round(inside, 2),
        "luma_margin": round(luma_margin, 1),
        "luma_contrast": round(contrast, 1),
        "area": float(mask.mean()),
        "bbox": box,
        "aspect": (round((box[3] - box[2] + 1) / max(box[1] - box[0] + 1, 1), 3)
                   if box else 0.0),
    }


# NOTE - a connected-component count was tried here and REMOVED. It looked like
# the obvious way to catch the duplicate-garment ghost: label the raw cue mask,
# call anything over ~6% of the main blob a second object. Measured against ten
# candidates classified by eye it was wrong in both directions.
#
#   * runs/20260827_104532 cand_04 is unmistakably two sweaters and it passed.
#     The copies OVERLAP, so they are one connected component. Component
#     counting cannot see a duplicate that touches the original, and most of
#     them do.
#   * the same run's cand_08 is a single clean garment and it failed, on three
#     phantom components at 0.18, 0.12 and 0.07 - backdrop gradient and shadow
#     picked up by the luminance cue.
#
# The geometry is not separable here. Counting garments is a semantic question
# and it is asked of the vision model instead, in grade_flats.count_garments().


def garment_mask(path: Path, long_side: int = 1024, cue: str | None = None):
    """Boolean mask of the garment, largest connected blob only.

    Returns (mask, working_size). See garment_evidence() for which cue found it
    and on what numbers; this wrapper exists because most callers only want the
    pixels.
    """
    e = garment_evidence(path, long_side, cue)
    return e["mask"], e["shape"]


def silhouette(path: Path, long_side: int = 1024, cue: str | None = None) -> dict:
    """Area and bounding box of the garment outline, on a fixed grid.

    clean.py measures this before and after cleaning. A step that is allowed to
    change pixels is not allowed to change the outline, and the bounding box is
    what makes that checkable: on the run this was written for, a good clean
    moved the top edge 6 px on a 787 px garment while a bad one moved it 99 px.
    Area alone does not separate those two - matting shaves 5% off a perfectly
    good outline just by trimming anti-aliased edge pixels.

    `cue` is threaded through, and the cue used comes back in the result: two
    silhouettes are only comparable if they were found the same way. A source
    measured on luminance against a cleaned copy measured on chroma reported
    "outline lost 81.3% of its area" for a clean that had lost nothing.
    """
    e = garment_evidence(path, long_side, cue)
    m, shape = e["mask"], e["shape"]
    grid = [int(shape[0]), int(shape[1])]
    if not m.any():
        return {"area": 0.0, "bbox": None, "grid": grid, "cue": e["cue"]}
    ys, xs = np.where(m)
    return {"area": float(m.mean()),
            "bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())],
            "grid": grid, "cue": e["cue"]}


def _pose_holes(mask: np.ndarray, min_frac: float = 0.015):
    """Keep the holes that describe the POSE, close the ones that are print.

    outline_map() keeps the mask's holes unfilled, and it has to: the gap
    between a bra's crossed straps and the gap between two legs are most of what
    makes a lay readable, and filling them turns the reference into a blob.

    But a hole is only a mask artefact, and anything the cue reads as
    not-garment becomes one. On the pullover reference the intarsia teddy bear
    is lighter than the navy body, so the luminance cue punched it out and the
    outline came back with a bear-shaped patch of speckles in the middle of the
    chest - construction, in the one image whose entire purpose is to carry no
    construction. Whatever else the model copies off the lay reference, it must
    not be a graphic.

    Area separates the two cleanly. A strap gap or a leg gap is a large share of
    the garment; a print, a logo or a light stripe is not. Everything under
    `min_frac` of the garment's own area is closed.

    Returns (mask, holes_closed).
    """
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask, 0
    lab, n = ndimage.label(holes)
    if n == 0:
        return mask, 0
    area = float(filled.sum())
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    small = [i + 1 for i, s in enumerate(sizes) if float(s) / area < min_frac]
    if not small:
        return mask, 0
    return (mask | np.isin(lab, small)), len(small)


def outline_map(src: Path, dst: Path, long_side: int = 2048,
                stroke: int = 3, hole_frac: float = 0.015) -> dict:
    """Write a line drawing of the garment's outline: pose, no construction.

    The lay reference exists to say how the garment should be ARRANGED - straps
    flat and symmetric, legs closed, hems level. It is not supposed to say
    anything about how the garment is BUILT, and the model has no way to tell
    those apart when it is handed a photograph of a different product. On
    runs/20260819_205617 the reference carried a defined V-neckline and seam
    piping that the real garment does not have, and four of ten candidates came
    back with a neckline seam and topstitching along the straps.

    An outline cannot leak construction because it does not contain any. The
    garment's own holes are kept - the gap between crossed straps is most of
    what makes the pose readable - and everything inside the fabric is gone.

    Returns what was written, for the record.
    """
    e = garment_evidence(src, long_side=1024)
    m = e["mask_open"]
    if not m.any():
        raise RuntimeError(f"no garment found in {src}, so no outline to draw")
    m, closed = _pose_holes(m, min_frac=hole_frac)

    W, H = Image.open(src).size
    k = long_side / max(W, H)
    W, H = max(1, int(W * k)), max(1, int(H * k))
    big = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                     .resize((W, H), Image.NEAREST)) > 127
    edge = big ^ ndimage.binary_erosion(big, np.ones((3, 3)),
                                        iterations=max(1, stroke))
    out = np.where(edge, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(dst, quality=95, subsampling=0)
    return {"size": [W, H], "cue": e["cue"], "stroke": stroke,
            "holes_closed": closed,
            "ink": round(float(edge.mean()) * 100, 3)}


def clean_verdict(run_dir: Path) -> dict:
    """What the pre-clean step concluded, read off <run>/archive/clean_audit.json.

    clean.py already prints "PRE-CLEAN FAILED" and exits 1 when the cleaned
    image is not the same garment as the source. On runs/20260819_205617 that
    happened, and the run generated ten images from the failed source anyway:
    the exit code went nowhere, prepare.py verified only that a file existed,
    and every step after it - the description, the prompt, the construction
    check - inherited a source the gate had already rejected. 150 cents.

    So the verdict is written down where anything about to spend money can read
    it. `ok` is False only when the audit says the gate failed; a missing or
    unreadable audit is `checked: False`, which is a different thing and must
    not be treated as a failure - clean.py is skippable on purpose.
    """
    p = Path(run_dir) / "archive" / "clean_audit.json"
    if not p.exists():
        return {"checked": False, "ok": True, "fails": [], "why": f"no {p.name}"}
    try:
        audit = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {"checked": False, "ok": True, "fails": [],
                "why": f"{p.name} unreadable ({type(e).__name__})"}
    attempts = audit.get("attempts") or []
    if not attempts:
        return {"checked": False, "ok": True, "fails": [],
                "why": f"{p.name} records no attempt"}
    last = attempts[-1]
    fails = list(last.get("outline_fails") or [])
    vision = last.get("vision") or {}
    missing = str(vision.get("missing", "none")).strip().lower()
    if vision and not vision.get("garment_intact", True) and missing not in ("", "none"):
        fails.append(f"the vision check says the clean lost {vision.get('missing')!r}")
    return {"checked": True, "ok": not fails, "fails": fails,
            "attempts": len(attempts),
            "why": "; ".join(fails) if fails else "outline held"}


def garment_box(path: Path, like: Path | None = None, long_side: int = 1024,
                area_ratio: float = 2.5, aspect_ratio: float = 1.6) -> dict:
    """The garment's bounding box in this image's own full-resolution pixels,
    with a fallback for when the box that came back is not credible.

    A crop tool that trusts the mask blindly produces a confident close-up of
    the wrong strip of fabric, and nothing downstream can tell: the crop looks
    like a crop, the judge answers the question it was asked, and the verdict is
    about a band when it was supposed to be about a neckline. That happened on
    runs/20260819_205617, where the mask on three candidates collapsed to a
    band-height sliver and the sheets showed close-ups nobody could compare.

    So when `like` is given - the cleaned source, normally - the measured box is
    checked against it in the only two terms that survive a re-lay and a
    reframe: how much of its own frame the garment fills, and its aspect. Wildly
    off either and the source's own fractional box is used instead, which is at
    worst approximately right rather than precisely wrong. `ok` says which
    happened and `note` says why, and callers are expected to print it.

    The cue is NOT forced to match `like`, and that is deliberate. clean.py does
    force it, because there the two images are the same photograph before and
    after and a cue that changes between them invents a difference. Here the
    other image is a fresh re-synthesis with its own plate and its own
    colourway, and the cue that reads it best is genuinely allowed to differ -
    forcing the source's cue onto runs/20260819_205617's candidates would put
    three of them back on the 3.2% shadow strip this check exists to catch.
    What travels across the pair is the CHECK below, not the method.
    """
    ref = garment_evidence(like, long_side) if like and Path(like).exists() else None
    e = garment_evidence(path, long_side)
    W, H = Image.open(path).size

    def scaled(box, shape):
        y0, y1, x0, x1 = box
        sx, sy = W / shape[1], H / shape[0]
        return (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))

    def fractional(box, shape):
        y0, y1, x0, x1 = box
        return (x0 / shape[1], y0 / shape[0], (x1 + 1) / shape[1], (y1 + 1) / shape[0])

    why = []
    if e["bbox"] and ref and ref["bbox"]:
        fa, fb = e["area"], ref["area"]
        ra = fa / fb if fb else 0.0
        rs = (e["aspect"] / ref["aspect"]) if ref["aspect"] else 0.0
        if not (1 / area_ratio <= ra <= area_ratio):
            why.append(f"fills {fa*100:.1f}% of frame against the source's "
                       f"{fb*100:.1f}% ({ra:.2f}x, limit {area_ratio:g}x)")
        if not (1 / aspect_ratio <= rs <= aspect_ratio):
            why.append(f"aspect {e['aspect']:.2f} against the source's "
                       f"{ref['aspect']:.2f} ({rs:.2f}x, limit {aspect_ratio:g}x)")

    if e["bbox"] and not why:
        return {"box": scaled(e["bbox"], e["shape"]), "ok": True, "cue": e["cue"],
                "note": (f"garment found on {e['cue']} "
                         f"({e['area']*100:.1f}% of frame, aspect {e['aspect']:.2f})"),
                "evidence": e}

    if ref and ref["bbox"]:
        u0, v0, u1, v1 = fractional(ref["bbox"], ref["shape"])
        box = (int(u0 * W), int(v0 * H), int(u1 * W), int(v1 * H))
        reason = "; ".join(why) or "no garment found"
        return {"box": box, "ok": False, "cue": e["cue"],
                "note": (f"WARNING: the box measured here is not credible "
                         f"({reason}). Using the source's own fractional box "
                         f"instead - approximately right, not measured."),
                "evidence": e}

    return {"box": (0, 0, W - 1, H - 1), "ok": False, "cue": e["cue"],
            "note": ("WARNING: no garment found and no source to fall back on; "
                     "using the whole frame."),
            "evidence": e}


def speck_count(path: Path, min_px: int = 24) -> int:
    """Blobs other than the garment, above min_px. Background dirt and hairlines."""
    im = Image.open(path).convert("L")
    im.thumbnail((1024, 1024), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    b = 8
    border = np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                             a[:, :b].ravel(), a[:, -b:].ravel()])
    mask = a < float(np.median(border)) - 18.0
    mask = ndimage.binary_opening(mask, np.ones((2, 2)))
    lab, n = ndimage.label(mask)
    if n <= 1:
        return 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    biggest = np.max(sizes)
    return int(((sizes >= min_px) & (sizes < biggest)).sum())


def rigid_dims(mask: np.ndarray) -> dict:
    """Dimensions that survive a legitimate re-lay of the garment.

    Overall bbox width is NOT one of them: closing a pair of splayed legs, or
    folding straps in, narrows the bbox on purpose. This project's own source
    measures 60.7% of frame width against a library range of 40.3-52.4%, so
    penalising that narrowing would penalise the fix.

    What genuinely cannot change is how long the garment is and how wide its top
    band is - a waistband or a bra band is rigid however the rest is arranged.
    Those two catch a garment squeezed narrower or stretched longer, which is a
    lie about the product, while letting the re-lay happen.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return {"length": 0.0, "top_width": 0.0, "solidity": 0.0}
    y0, y1 = ys.min(), ys.max()
    length = float(y1 - y0 + 1)

    # Top band: the widest run across the top eighth of the garment.
    band = mask[y0:y0 + max(1, int(length * 0.125))]
    per_row = band.sum(axis=1)
    top_width = float(per_row.max()) if per_row.size else 0.0

    # Solidity: how much of its own convex hull the silhouette fills. Drops when
    # limbs splay, rises when they close - a lay-quality signal, not a gate.
    area = float(mask.sum())
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack((xs, ys))
        step = max(1, len(pts) // 4000)
        hull = float(ConvexHull(pts[::step]).volume)
        solidity = area / hull if hull else 0.0
    except Exception:
        solidity = 0.0

    return {"length": length, "top_width": top_width, "solidity": solidity}


def clipped(mask: np.ndarray, margin_px: int = 2) -> str:
    """Which frame edges the garment touches.

    The one framing fault a retoucher cannot repair: pixels that were never
    captured. Everything else about position is a transform on a layer.
    """
    H, W = mask.shape
    hit = []
    if mask[:margin_px + 1].any():
        hit.append("top")
    if mask[-(margin_px + 1):].any():
        hit.append("bottom")
    if mask[:, :margin_px + 1].any():
        hit.append("left")
    if mask[:, -(margin_px + 1):].any():
        hit.append("right")
    return ",".join(hit)


def soft_alpha(path: Path, feather: float = 1.2) -> np.ndarray:
    """Soft alpha for a cutout, at the image's own resolution.

    A binary mask cuts a hard, aliased edge that reads as a paste-up. This ramps
    alpha over the plate-to-garment transition instead, so the edge keeps its
    natural softness, then restricts to the garment blob so background specks do
    not come along.
    """
    im = Image.open(path).convert("L")
    a = np.asarray(im, dtype=np.float32)
    b = 8
    border = np.concatenate([a[:b].ravel(), a[-b:].ravel(),
                             a[:, :b].ravel(), a[:, -b:].ravel()])
    plate = float(np.median(border))

    # Ramp alpha across the FULL plate-to-garment contrast, not a fixed slice.
    # A fixed 16-level ramp captured only the top sliver of an edge that falls
    # from 247 to about 60, so genuinely half-covered pixels came out fully
    # opaque and carried plate colour into the cutout as a light fringe.
    core = ndimage.binary_erosion(a < plate - 25.0, np.ones((7, 7)))
    fg = float(np.percentile(a[core], 40)) if core.sum() > 200 else plate - 60.0
    span = max(plate - fg, 30.0)          # a pale garment needs a floor here
    alpha = np.clip((plate - a) / span, 0.0, 1.0)

    solid = alpha > 0.5
    solid = ndimage.binary_fill_holes(ndimage.binary_opening(solid, np.ones((3, 3))))
    lab, n = ndimage.label(solid)
    if n:
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        keep = lab == int(np.argmax(sizes)) + 1
        # Dilate before masking so the soft ramp outside the solid core survives.
        alpha *= ndimage.binary_dilation(keep, np.ones((9, 9)))
        # The interior must be fully opaque. Normalising the ramp against a
        # representative garment tone leaves lighter areas of the garment at
        # alpha ~0.85, which lets the background bleed through the middle of the
        # product. Only the boundary band should ever be partial.
        alpha = np.maximum(alpha, ndimage.binary_erosion(
            keep, np.ones((5, 5))).astype(np.float32))
    if feather:
        alpha = ndimage.gaussian_filter(alpha, feather)
    return np.clip(alpha, 0.0, 1.0)


def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbour resample of a boolean mask to a common canvas."""
    src = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(src.resize((shape[1], shape[0]), Image.NEAREST)) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def shape_stats(mask: np.ndarray) -> dict:
    """Tilt, centroid offset and mirror symmetry of one silhouette."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return {"tilt": 0.0, "cx": 0.0, "cy": 0.0, "symmetry": 0.0}
    H, W = mask.shape

    # Tilt: how far the silhouette's principal axis leans off the NEAREST frame
    # axis, not off vertical.
    #
    # Measuring off vertical only works for garments taller than they are wide.
    # Every bra in this project's library is wider than tall, so its major axis
    # is horizontal and the off-vertical figure reads +-88 degrees on a perfectly
    # level laydown - which would reject 100% of them at a 3-degree bar. Folding
    # to the nearest axis also makes the number stable for near-square garments
    # (several bras measure aspect 0.98-1.03), where the major axis can flip
    # between horizontal and vertical on a pixel.
    x, y = xs - xs.mean(), ys - ys.mean()
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    axis = float(np.degrees(np.arctan2(vx, vy)))
    if axis > 90:
        axis -= 180
    if axis < -90:
        axis += 180
    tilt = ((axis + 45) % 90) - 45      # lean from whichever axis is nearer

    # Symmetry: IoU against the horizontal mirror of the silhouette's own bbox,
    # so a garment that is symmetric but off-centre still scores well - centring
    # is measured separately below.
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = mask[y0:y1, x0:x1]
    sym = iou(crop, crop[:, ::-1])

    return {"tilt": tilt, "axis": axis,
            "cx": float((xs.mean() - W / 2) / W * 100),
            "cy": float((ys.mean() - H / 2) / H * 100),
            "symmetry": float(sym)}


def garment_rgb(path: Path, mask: np.ndarray) -> np.ndarray:
    """Mean RGB inside the mask, at the mask's working resolution."""
    im = Image.open(path).convert("RGB")
    im = im.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    if not mask.any():
        return np.zeros(3, dtype=np.float32)
    return a[mask].mean(axis=0)


def bbox_crop(mask: np.ndarray) -> np.ndarray:
    """The mask cropped to its own bounding box."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return mask
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def norm_mask(mask: np.ndarray, grid: int = 256) -> np.ndarray:
    """The silhouette on a fixed square grid, cropped to its own bbox first.

    Two images of the same garment rarely share a coordinate system - this
    project's source is 4:3 and its candidates are 3:4, and the generator
    rescales and re-centres on purpose - so comparing raw masks compares
    framing. Normalising to the bbox throws away exactly the framing the
    retouch team sets itself and keeps the shape, which is the thing that has
    to survive a re-lay.
    """
    if not mask.any():
        return np.zeros((grid, grid), dtype=bool)
    crop = bbox_crop(mask)
    im = Image.fromarray((crop * 255).astype(np.uint8))
    return np.asarray(im.resize((grid, grid), Image.BILINEAR)) > 127


def silhouette_iou(a: np.ndarray, b: np.ndarray, grid: int = 256) -> float:
    """Bbox-normalised IoU of two silhouettes, 0-1.

    Measured on runs/20260819_205617: candidates that were genuinely re-laid
    scored 0.85-0.88 against the cleaned source, and the ones a vision pass and
    a human both called redrawn scored 0.77-0.81. That gap is the whole reason
    this metric exists - no photometric measurement in the pipeline separated
    those two groups at all, and the old grade ranked them the wrong way round.
    """
    return iou(norm_mask(a, grid), norm_mask(b, grid))


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """One sRGB triple (0-255) to CIE L*a*b*, D65. Small enough to keep here
    rather than take a dependency on skimage for three lines of matrix."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (m @ c) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def colour_drift(rgb_a: np.ndarray, rgb_b: np.ndarray) -> dict:
    """How far apart two garment colours are: mean channel difference and dE76.

    dE is the one to score on - it is perceptual, so 1.0 is invisible and 5.0 is
    a different colourway - while the raw RGB difference stays in the record
    because it is the number anyone can check by eye against the pixels.
    """
    a = np.asarray(rgb_a, dtype=np.float64)
    b = np.asarray(rgb_b, dtype=np.float64)
    return {"drgb": float(np.abs(a - b).mean()),
            "de": float(np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b)))}


# A numeric wrinkle metric was tried and removed once, and the reason is worth
# keeping: creases are broad, soft, oriented ridges, and an isotropic band-pass
# at every scale from sigma 3-12 to 2-60 ranked the visibly SMOOTHEST candidate
# of a real run highest, because it was reading the garment's form shading
# rather than its creases.
#
# The measurement below is the same class of statistic and inherits that flaw -
# so it is deliberately NOT used as "lower is better". It is used as a distance
# from the SOURCE's own value. Form shading appears in both images and cancels;
# what is left is a candidate whose surface is rougher than the real garment
# (creases the re-lay failed to relax) or smoother than it (a redraw that
# ironed the knit out of existence). Ranked that way round, the metric answers
# a question it can actually answer.
def wrinkle_energy(path: Path, mask: np.ndarray, norm_h: int = 900,
                   win: int = 9, long_side: int = 1024) -> float:
    """Mean local luminance SD inside the garment, at a fixed garment height.

    Rescaled so the garment's bbox is `norm_h` tall before anything is
    measured, or the number mostly reports source resolution: a 3072px original
    resampled to a common width is smoother than a 1792px one at the same
    width, purely from the heavier resample.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    k = norm_h / max(ys.max() - ys.min() + 1, 1)
    L = Image.open(path).convert("L")
    L.thumbnail((long_side, long_side), Image.LANCZOS)
    w, h = int(L.width * k), int(L.height * k)
    a = np.asarray(L.resize((w, h), Image.LANCZOS), dtype=np.float32)
    m = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                   .resize((w, h), Image.NEAREST)) > 127
    inner = ndimage.binary_erosion(m, np.ones((15, 15)))
    if inner.sum() < 500:
        inner = m
    if not inner.any():
        return 0.0
    mean = ndimage.uniform_filter(a, win)
    sq = ndimage.uniform_filter(a * a, win)
    return float(np.sqrt(np.clip(sq - mean * mean, 0.0, None))[inner].mean())


def seam_energy(path: Path, mask: np.ndarray) -> float:
    """Laplacian energy well inside the silhouette.

    Run against the source's own value this reads as invented detail: a model
    that hallucinates topstitching drives it up, one that renders cloudy paper
    instead of knit drives it down. Eroded hard so the silhouette edge itself
    never contributes.
    """
    im = Image.open(path).convert("L")
    im = im.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    inner = ndimage.binary_erosion(mask, np.ones((9, 9)))
    if inner.sum() < 100:
        inner = mask
    if not inner.any():
        return 0.0          # .std() of an empty slice is nan, which reads as a
                            # number in the table and is not one
    e = ndimage.laplace(ndimage.gaussian_filter(a, 1))
    return float(e[inner].std())


# --------------------------------------------------------------------------
# Prompt guardrails
#
# One definition, imported by promptfile.py (for prompt_show) and generate.py
# (for the refusal). They used to live in generate.py alone, which meant the
# only way to discover a rule was to hit it with money on the line.
#
# Two tiers, and the split is deliberate:
#
#   PROBLEMS refuse the run. Each one is a mechanism that provably corrupts the
#   output - not a matter of taste - and --force is the only way past.
#
#   WARNINGS print and generate anyway. These are judgement calls. The old
#   pipeline refused on the word count and the must-mention list too, because
#   the prompt was written blind in one shot and never seen again. The prompt is
#   now sectioned and inspectable for free with prompt_show, so a short prompt is
#   a thing the model can see and decide about rather than a wall.
# --------------------------------------------------------------------------

# Asking for transparency produces a painted checkerboard: the endpoint cannot
# output an alpha channel, and a real run came back with a literal grey-and-white
# checker pattern counted as 1587-3135 background specks.
FORBID_TRANSPARENT = ("transparent", "transparency", "alpha channel",
                      "remove the background", "background removed",
                      "removed background", "no background")

# Paperwork and fasteners: still a warning, but the reason changed.
#
# It is now handled by a STANDING CLAUSE that generate.py appends to every
# prompt - "any pin, clip, tack, hanger, hook, price ticket or swing tag that was
# holding it in place for the shot is gone", paired with "everything sewn into
# the garment stays exactly as it is". It had to become standing because
# segment.py drops the BACKGROUND and nothing else, so these survive into image 1
# every time and nothing else removes them - and on runs/20260827_223611 not one
# prompt section mentioned them, so they shipped.
#
# So the agent no longer needs to ask, and asking again is the risk this warning
# now guards. Repeating a standing instruction in the agent's own words is how
# the two end up disagreeing about what counts as a label: the clause carefully
# separates the temporary fastening (goes) from the sewn-in label and logo
# (stays), and a second, looser sentence about "labels" undoes that distinction.
# Naming a thing repeatedly is also how a diffusion model is told to draw it -
# the measured case was a prompt that described a hang tag and asked it to STAY
# IN PLACE, and four candidates grew one that had never been sent.
WARN_PAPERWORK = ("hang tag", "hangtag", "swing tag", "price ticket",
                  "ticket", "barcode", "hanger", "tag", "label", "pin",
                  "pins", "clip", "clips", "tack", "tacks")

# Naming which FACE of the garment is shown is how it comes back flipped: one
# batch opened "the garment is shown from the back" and seven of ten candidates
# returned the reverse face, every seam correct and the garment inside out.
#
# The rule used to fire on the bare phrase "viewed from", which is too wide.
# "flat lay viewed from directly above" is the correct camera for this job and
# has nothing to do with which face is up. So a viewpoint phrase is only a
# problem when it carries a FACE word, and the overhead vocabulary is allowed
# outright.
FACE_WORDS = ("back", "front", "rear", "reverse", "side", "underside",
              "inside", "inside out", "wrong side", "three-quarter",
              "three quarter", "behind")

OVERHEAD_WORDS = ("above", "directly above", "from above", "overhead",
                  "top-down", "top down", "birds eye", "bird's eye",
                  "plan view", "straight down", "looking down")

# The phrasings that introduce a viewpoint. Each is checked for a FACE_WORD
# within the window that follows it.
#
# Every lead here is unambiguous camera language. A bare "from the" was tried and
# removed: it turns ordinary construction wording into a refusal - "the placket
# runs from the collar to the hem, with the front band ribbed" has "from the"
# and "front" eleven words apart and means nothing about which face is up. A
# false refusal is worse than a missed one, because the skill already tells the
# model not to name a face; this is the backstop, not the instruction.
VIEWPOINT_LEADS = ("viewed from", "shown from", "seen from", "photographed from",
                   "shot from", "taken from", "rendered from", "captured from")

# ...and the compact forms that are a viewpoint all by themselves.
VIEWPOINT_PHRASES = ("back view", "front view", "rear view", "side view",
                     "reverse view", "reverse side", "wrong side", "inside out",
                     "from behind", "three-quarter view", "three quarter view",
                     "flipped over", "turned over", "opposite side",
                     "the other side")

VIEWPOINT_WINDOW = 40      # chars after a lead in which a face word counts

# Absolutes that ask for MORE than flat: they push the model past relaxing the
# handling folds and into repainting the fabric as a smooth surface, which is
# this project's documented redraw driver. A warning, not a refusal - the
# wording is a judgement call, and metrics.py measures the consequence directly
# as a wrinkle ratio that falls too far.
SOFTEN = ("wrinkle-free", "wrinkle free", "no creases", "no wrinkles",
          "completely smooth", "perfectly smooth", "freshly steamed",
          "steamed and pressed", "ironed", "no fold lines", "flawless")

# What a prompt is expected to say. Advisory - see the tier note above.
MUST_MENTION = (
    ("flatness", ("flat", "flatly", "flatness", "laid flat", "lay flat"),
     "the model adds volume and 3D shaping unless told not to"),
    ("wrinkles", ("wrinkle", "crease", "rumple", "fold line", "steamed",
                  "pressed"),
     "de-wrinkling is the job; it does not happen by default"),
    ("the background", ("background", "backdrop", "plate", "white studio"),
     "an unmentioned backdrop comes back grey, textured or replaced"),
)

# Checked separately from MUST_MENTION, and with WORD BOUNDARIES, because the
# obvious way is broken. MUST_MENTION matches substrings on purpose - "flat" has
# to catch "flatly" and "flatness" - but a substring test for the limbs matches
# "arm" inside "garment", and every prompt in this project says "garment". The
# guard would have been permanently satisfied and never once fired.
#
# It exists because the sleeve angle came out of the appended clause: that clause
# used to state a pose for every garment, so a prompt that never mentioned
# sleeves still got one - the wrong one, whenever the reference disagreed. Now
# the clause is silent on the angle, and a prompt that does not name it leaves
# the most visible thing about a laydown to chance.
LIMB_WORDS = ("sleeve", "sleeves", "cuff", "cuffs", "arm", "arms",
              "leg", "legs")

LIMB_WHY = ("the standing clause sets symmetry and level, not the ANGLE. Read it "
            "off image 2 and say it here - splayed wide away from the body, "
            "angled down at 45 degrees, hanging straight at the sides - and say "
            "how far the cuffs sit from the body. Nothing else decides it.")

MIN_WORDS = 120

GREYSCALE_WORDS = ("greyscale", "grayscale", "grey", "gray", "tone",
                   "colourless", "colorless", "outline", "silhouette",
                   "line drawing", "desaturated", "black and white")


def _overhead_span(low: str, at: int) -> bool:
    """Is the viewpoint starting at `at` an overhead one?

    Checked before any face match so the whitelist wins outright. The window is
    generous on purpose: "viewed from directly above the garment" puts three
    words between the lead and the whitelist term.
    """
    window = low[at:at + VIEWPOINT_WINDOW]
    return any(w in window for w in OVERHEAD_WORDS)


def viewpoint_hits(prompt: str) -> list[str]:
    """Face-naming viewpoint phrases in `prompt`, overhead ones excluded."""
    low = prompt.lower()
    hits: list[str] = []

    for phrase in VIEWPOINT_PHRASES:
        if phrase in low:
            hits.append(phrase)

    # Several leads can point at the same face word - "is shown from the back"
    # matches both "shown from" and, on an earlier draft of this list, "from
    # the". Report the offending spot once, keyed by where the face word sits.
    claimed: set[int] = set()
    for lead in VIEWPOINT_LEADS:
        start = 0
        while True:
            at = low.find(lead, start)
            if at < 0:
                break
            start = at + len(lead)
            if _overhead_span(low, at):
                continue                    # "viewed from directly above" - fine
            window = low[at:at + VIEWPOINT_WINDOW]
            m = next((m for f in FACE_WORDS
                      if (m := re.search(rf"\b{re.escape(f)}\b", window))), None)
            if m and (at + m.start()) not in claimed:
                claimed.add(at + m.start())
                hits.append(f"{lead} ... {m.group(0)}")

    seen, out = set(), []
    for h in hits:                          # stable order, no duplicates
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def check_prompt(prompt: str, reference: Path | None = None,
                 min_words: int = MIN_WORDS) -> tuple[list[str], list[str]]:
    """(problems, warnings). Problems refuse the run; warnings only print.

    `reference` is optional: when it is a greyscale image, a prompt that never
    says so is warned about, because read as a colour target it desaturates the
    garment.
    """
    low = prompt.lower()
    problems: list[str] = []
    warnings: list[str] = []

    # ---- problems ---------------------------------------------------------
    hits = [w for w in FORBID_TRANSPARENT if w in low]
    if hits:
        problems.append(
            f"asks for a transparent or removed background "
            f"({', '.join(hits)}) - the endpoint cannot output an alpha channel "
            f"and paints a checkerboard into the pixels instead. Ask for a plain "
            f"white plate.")

    hits = viewpoint_hits(prompt)
    if hits:
        problems.append(
            f"names which face of the garment is shown ({', '.join(hits)}) - "
            f"this is how it comes back flipped, and seven of ten candidates in "
            f"one batch did. Say nothing about sides; the appended clause already "
            f"asks for the same face as image 1. Overhead camera wording - "
            f"'viewed from directly above', 'top-down' - is fine and is not this.")

    # ---- warnings ---------------------------------------------------------
    hits = [w for w in WARN_PAPERWORK
            if re.search(rf"\b{re.escape(w)}\b", low)]
    if hits:
        warnings.append(
            f"names paperwork or fasteners ({', '.join(hits)}) - this is already "
            f"handled. Every prompt carries a standing clause saying that any "
            f"pin, clip, tack, hanger, hook, ticket or swing tag holding the "
            f"garment up is gone from the finished photograph, and that "
            f"everything SEWN IN - seams, stitching, brand and care labels, "
            f"embroidery, logos - stays exactly as it is. Delete your version: "
            f"two sets of words about labels is how the sewn-in ones get removed "
            f"along with the temporary ones. Say something here only if this "
            f"garment has an unusual case the clause cannot know about.")

    n = len(prompt.split())
    if n < min_words:
        warnings.append(
            f"{n} words, under the {min_words}-word guide. Short prompts leave "
            f"the lay, the flatness and the construction to the model's own "
            f"judgement, which is the thing being replaced.")

    for name, words, why in MUST_MENTION:
        if not any(w in low for w in words):
            warnings.append(f"never mentions {name} - {why}")

    if not any(re.search(rf"\b{w}\b", low) for w in LIMB_WORDS):
        warnings.append(f"never says where the sleeves or legs go - {LIMB_WHY}")

    hits = [w for w in SOFTEN if w in low]
    if hits:
        warnings.append(
            f"asks for absolute smoothness ({', '.join(hits)}). Relaxing handling "
            f"folds is the job; ironing the knit out of existence is a redraw, and "
            f"the wrinkle ratio falls just as far in that direction. Prefer "
            f"'relax the folds so the fabric lies flat, keep the knit's real "
            f"texture'.")

    if reference is not None:
        try:
            greyscale = Image.open(reference).mode == "L"
        except Exception:       # noqa: BLE001 - an unreadable reference fails later
            greyscale = False
        if greyscale and not any(w in low for w in GREYSCALE_WORDS):
            warnings.append(
                "never says image 2 is greyscale - read as a colour target it "
                "desaturates the garment.")

    return problems, warnings


def format_findings(problems: list[str], warnings: list[str]) -> str:
    """One rendering of check_prompt's output, so every caller reads the same."""
    lines = []
    if problems:
        lines.append(f"{len(problems)} problem(s) - these refuse the run:")
        lines += [f"  REFUSE  {p}" for p in problems]
    if warnings:
        lines.append(f"{len(warnings)} warning(s) - printed, not blocking:")
        lines += [f"  warn    {w}" for w in warnings]
    if not lines:
        lines.append("no problems, no warnings.")
    return "\n".join(lines)
