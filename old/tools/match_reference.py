#!/usr/bin/env python3
"""
Find the library reference image that best matches a query garment photo,
using a self-hosted Qwen-VL model behind an OpenAI-compatible API.

Any garment category, not just bras: see "Library layout" below.

Pipeline
  A. One VLM call per image -> strict JSON attribute record (neckline, straps,
     band, fabric, color hex, ...). Cached on disk, so re-runs are ~free.
     The QUERY is described first, because its garment_type is what selects
     the library category (see below).
  B. Deterministic weighted scoring of each library record against the query
     record, in plain Python. Colour distance via CIE Lab deltaE.
  C. One final VLM call showing the query + top-K side by side, to pick a
     winner and say why. This is a check on B, not a replacement for it.

Library layout
  Flat (all .jpg in one folder) is used as-is. If the library instead holds
  category subfolders - library_reference/{bras,leggings} - the query's own
  garment_type picks the folder automatically, so a bra query is never scored
  against leggings. Override with --category, inspect with --list-categories.

Stdlib only - no pip installs. Image prep uses macOS `sips` and ImageMagick
`magick`, both already on this machine.

Usage
  python3 match_reference.py --check                     # probe server, list models
  python3 match_reference.py                             # full run with defaults
  python3 match_reference.py --top-k 5 --color-weight 0  # rank on cut/shape only
  python3 match_reference.py --base-url http://127.0.0.1:8080/v1

VENDORED into PLD_Harness from ~/Desktop/SELECTING_REFERENCE. Four deliberate
changes from upstream, all of them path/plumbing, so an upstream diff stays
readable:

  * ROOT is the PROJECT root (the parent of tools/), not this file's folder, so
    --query and --library default to the harness's own inputs/ and
    library_reference/ rather than to tools/.
  * the cache lives in .cache/refmatch/, not .cache/ - the harness already keeps
    a .cache/small for a different tool's downscales.
  * --out-dir controls where match_results.json and result_top_matches.jpg land,
    so the harness can drop them in the run folder instead of the project root.
  * the base URL comes from REFMATCH_BASE_URL, NOT QWEN_BASE_URL. The harness
    uses QWEN_BASE_URL for its own text model and appends /v1 itself; this file
    expects the /v1 already on the URL. Sharing one variable pointed the matcher
    at the wrong port with the wrong path shape.

This file is the matcher only. Installing the winner into inputs/ is
select_reference.py, which drives this one.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # the project root, not tools/
CACHE = ROOT / ".cache" / "refmatch"
SMALL = CACHE / "small"
ATTRS = CACHE / "attrs"


def _default_base_url() -> str:
    """The VISION server, which is not necessarily the harness's text server.

    Deliberately does not read QWEN_BASE_URL: harness.py reads that one and
    appends /v1 itself, while everything here concatenates onto the URL as
    given. One shared variable meant one of the two was always wrong.
    """
    url = os.environ.get("REFMATCH_BASE_URL", "http://10.11.245.41:8091/v1").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


DEFAULT_BASE_URL = _default_base_url()
DEFAULT_MODEL = os.environ.get("REFMATCH_MODEL", "")  # empty = auto-detect from /v1/models


def _load_api_key() -> str:
    """QWEN_API_KEY, else a .qwen_key file next to this script. The server is
    llama.cpp started with --api-key: /v1/models is open but completions 401."""
    if os.environ.get("QWEN_API_KEY"):
        return os.environ["QWEN_API_KEY"]
    kf = ROOT / ".qwen_key"
    if kf.exists():
        return kf.read_text().strip()
    return "not-needed"


API_KEY = _load_api_key()

# Bump when the attribute prompt changes, to invalidate cached records.
# v3: garment_type gained "pullover" and "fleece", fabric_finish gained "fleece"
# and "sherpa". Every v2 record predates those words, so a cached fleece reads
# "other"/"textured" - which resolves to no category and no region profile at
# all. They have to be re-described, not merged with the new ones.
# v4: garment_type gained "jeans" and "pants", and the bottoms now carry a rule
# for telling the four of them apart. "shorts" was the only woven bottom in the
# schema, so a full-length woven leg had nowhere else to go: on
# runs/20260827_131825 a pair of wide-leg kids' jeans came back "shorts", which
# matches no folder, which pooled the whole library and ranked a pair of jeans
# against 39 sports bras. Every v3 record of a woven bottom is wrong in that
# same direction and has to be re-described rather than merged.
# v5: "closure" gained "unknown". The bottoms rule above has always said to
# answer "unknown" for closure, but the schema line offered only the four bra
# values, so the model could not comply and picked the nearest-sounding one -
# differently for each photo of the same fly. On runs/20260827_133232 a pair of
# jeans read "front_zip" and a near-identical library pair read "front_hook",
# scoring 0.0 on a field worth 1.0 of a 9.5 total: 88.7 against a threshold of
# 95, where dropping the field scores 99.2. Two of the seven bra fields are
# skipped for a bottom by being unknown on both sides, so one bogus zero is a
# large fraction of what is left to disagree about.
# v6: garment_type gained "boyfriend", the gathered-waist bottom.
#
# The category was asked for as "like pants but for kids", which is not a thing
# a classifier can see: these are flat lays on white with no scale reference in
# frame, and a toddler's jean photographed alone is indistinguishable from an
# adult's. So the rule keys on what IS visible, and the library picked it out
# rather than a guess - described under v5, 13 of the 14 assets in boyfriend/
# came back with "ruffled elasticated waistband", "elasticated waistband" or
# "elasticated smocked waistband" in their own notes, and neither asset in
# loose/ did. The gathered waist is the signal; "for kids" is a fact about the
# folder, not about the pixels.
#
# v7: the same rule, rebalanced. v6 phrased it as a test that "OVERRIDES jeans
# and pants", and the model duly made boyfriend the default: it took all 14 of
# boyfriend/, which is right, and both of loose/ as well, including the
# wide-leg button-fly jean it had itself described as flat-banded one version
# earlier. An override invites the model to reach for the overriding answer. So
# the two waistbands are now stated as one fork with both branches described in
# equal detail, and the leg shape - the thing that makes a boyfriend LOOK like a
# boyfriend, and the thing most likely to drag a flat-banded wide-leg across -
# is explicitly ruled out as evidence.
#
# v8: garment_type gained "cardigan", the sleeved knit that opens all the way
# down the centre front. The pullover/fleece rule was rewritten into an ordered
# pair rather than a third branch bolted on, because the two splits are not on
# the same axis: fleece is decided on FABRIC and cardigan on the FRONT, and a
# full-zip fleece jacket satisfies both. Pile is asked first and wins, which
# keeps every existing fleece where it was. Only then does the front decide.
PROMPT_VERSION = "v8"

IS_TTY = sys.stdout.isatty()


def transient(msg: str) -> None:
    """In-flight status. Only drawn on a terminal, where settled() overwrites it;
    piping or redirecting drops it so logs are not littered with half-lines."""
    if IS_TTY:
        print(msg, end="", flush=True)


def settled(msg: str) -> None:
    """Final line, replacing any transient status that preceded it."""
    print(("\r" + msg + " " * 25) if IS_TTY else msg)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def http_json(url: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class Client:
    def __init__(self, base_url: str, model: str, timeout: int = 300, think: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Qwen3 is a reasoning model: with thinking on, llama.cpp puts the chain
        # in `reasoning_content` and leaves `content` empty until it finishes.
        # Thinking measured ~8x slower here for no gain on this task, so it is off
        # by default, but the plumbing handles both.
        self.think = think

    def models(self) -> list[str]:
        out = http_json(f"{self.base_url}/models", timeout=20)
        return [m["id"] for m in out.get("data", [])]

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        ids = self.models()
        if not ids:
            raise RuntimeError("server returned no models")
        self.model = ids[0]
        return self.model

    def chat(self, content: list, max_tokens: int = 900, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # Thinking needs headroom for the chain *plus* the answer; running out
            # mid-chain returns an empty content field and finish_reason "length".
            "max_tokens": max_tokens * 4 if self.think else max_tokens,
            "temperature": temperature,
        }
        if not self.think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        out = http_json(f"{self.base_url}/chat/completions", payload, timeout=self.timeout)
        choice = out["choices"][0]
        msg = choice["message"]
        text = (msg.get("content") or "").strip()
        if not text and choice.get("finish_reason") == "length":
            raise RuntimeError(
                "model hit the token limit while thinking and returned no answer; "
                "raise --max-tokens or drop --think"
            )
        return text


# --------------------------------------------------------------------------
# Image prep
# --------------------------------------------------------------------------

def ensure_small(src: Path, max_dim: int = 1024) -> Path:
    """Downscale into .cache/small. The query file is 5464x8192 / 23MB; sending
    that raw would blow up both the request and the model's vision budget."""
    SMALL.mkdir(parents=True, exist_ok=True)
    # The parent folder is part of the cache name: with category subfolders two
    # different images can share a stem (bras/black.jpg, leggings/black.jpg) and
    # would otherwise silently overwrite each other's downscale.
    tag = hashlib.sha1(str(src.resolve().parent).encode()).hexdigest()[:8]
    dst = SMALL / f"{src.stem}__{tag}__{max_dim}.jpg"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    subprocess.run(
        ["sips", "-Z", str(max_dim), str(src), "--out", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def image_part(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def text_part(s: str) -> dict:
    return {"type": "text", "text": s}


# --------------------------------------------------------------------------
# Stage A - attribute extraction
# --------------------------------------------------------------------------

ATTR_PROMPT = """You are cataloguing women's intimates/activewear product photos.

Look at the garment in this image and fill in the schema below. Judge only what
you can actually see. If a field is genuinely not visible, use "unknown".
Ignore any hangtag, label, price ticket, background, shadow or prop - describe
the garment only.

If the garment is a BOTTOM (leggings, shorts, jeans, pants), the upper-body
fields do not apply: set neckline, strap_style, strap_width, adjusters, padding,
closure and support_level to "unknown" rather than guessing. Fields that are
"unknown" on both sides of a comparison are dropped from scoring, so this costs
you nothing.

"closure" in particular: the four closure values describe how an UPPER BODY
garment is put on. A fly, a button, a zip, a drawstring or an elastic waist on a
bottom is none of them, so answer "unknown" for every bottom. Do not reach for
the nearest-sounding value.

Separate the bottoms on LENGTH and CUT. Judge length against the KNEE, not
against the ankle: answer "shorts" only when the leg ends at or above the knee.
Everything ending below the knee is full length here, including a cropped, wide,
barrel or ankle-length leg. Among the full-length ones, answer "leggings" when
the leg is knitted and close-fitting to the leg, "jeans" for a woven denim
bottom, and "pants" for any other woven bottom - chino, cargo, jogger, trouser -
however wide or tapered the leg.

Then split the full-length WOVEN bottoms once more, on the WAISTBAND. Look at
the band itself and decide which of these two you can actually see:

  GATHERED - the fabric is drawn up along the band into ripples or folds, by
  elastic, shirring or smocking, and/or a paperbag frill stands up above the
  closure. Answer "boyfriend". Denim, corduroy, canvas and coated all qualify,
  and so does any leg: straight, barrel, wide or flared.

  FLAT - the band lies smooth and untucked all the way across, normally with
  belt loops and a fly or button front below it. Answer "jeans" for denim and
  "pants" for anything else woven.

Judge this on the band and nothing else. A wide, barrel, slouchy or flared leg
sits on a flat band just as often as on a gathered one, so the leg tells you
nothing here; neither does the fabric, and neither does how big the garment
looks. If the band lies smooth, it is "jeans" or "pants" however relaxed the
rest of the garment is.

If the garment is SLEEVED (pullover, fleece, cardigan), the bra fields do not
apply the same way: set strap_style, strap_width, adjusters, padding and
support_level to "unknown". Neckline, closure, band, coverage and fabric_finish
still apply.

For a sleeved garment, ask the two questions in this order.

FIRST, the FABRIC, which separates "fleece" from everything else sleeved. Answer
"fleece" when the surface is a raised pile that stands off the seams: fleece,
polar fleece, microfleece, sherpa, teddy. A pile is "fleece" whatever its front
does, zipped right down or not. A brushed INSIDE with a flat outer face is not a
pile - judge the face you can see.

SECOND, if it is not a pile, the CENTRE FRONT, which separates "cardigan" from
"pullover":

  OPENS THE WHOLE WAY - two separate front panels meeting down the middle, held
  by buttons, snaps or a zip running from neck to hem, or simply hanging open.
  Answer "cardigan". A hood, a collar, a pocket or any knit or stitch - cable,
  pointelle, jacquard, ribbed, flat - makes no difference.

  DOES NOT - one unbroken front panel, or a placket, half-zip or quarter-zip
  that stops somewhere above the hem, so the garment still has to go over the
  head. Answer "pullover", for a sweatshirt, sweater or hoodie in jersey, french
  terry or knit, however heavy.

The test is whether the front separates all the way to the hem. Buttons that run
only partway down are a pullover placket, not a cardigan.

Return ONE JSON object, nothing else. No prose, no markdown fence.

{
  "garment_type":  "sports_bra" | "bralette" | "t_shirt_bra" | "tank_top" | "leggings" | "shorts" | "jeans" | "pants" | "boyfriend" | "pullover" | "cardigan" | "fleece" | "other",
  "neckline":      "square" | "scoop" | "v_neck" | "sweetheart" | "high_neck" | "plunge" | "other",
  "strap_style":   "wide_straight" | "thin_straight" | "crossback" | "racerback" | "v_back" | "strappy_multi" | "halter" | "other",
  "strap_width":   "thin" | "medium" | "wide",
  "adjusters":     true | false,
  "band":          "wide_elastic" | "narrow_elastic" | "logo_band" | "smooth_no_band" | "other",
  "fabric_finish": "smooth_matte" | "shiny" | "ribbed" | "textured" | "fleece" | "sherpa" | "lace" | "mesh" | "printed",
  "padding":       "molded_cups" | "removable_pads" | "unpadded" | "unknown",
  "closure":       "pullover" | "front_zip" | "front_hook" | "back_hook" | "unknown",
  "support_level": "low" | "medium" | "high" | "unknown",
  "coverage":      "cropped_short" | "standard" | "longline",
  "color_name":    "<plain english, e.g. periwinkle blue>",
  "color_hex":     "#RRGGBB",
  "notes":         "<one short sentence on the single most distinctive feature>"
}"""

FIELDS = [
    "garment_type", "neckline", "strap_style", "strap_width", "adjusters",
    "band", "fabric_finish", "padding", "closure", "support_level", "coverage",
]


def parse_json_blob(text: str) -> dict:
    """Models wrap JSON in fences or prose often enough that this is worth it."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def attr_cache_path(img: Path, model: str, think: bool) -> Path:
    key = (f"{img.resolve()}|{img.stat().st_mtime_ns}|{model}"
           f"|{PROMPT_VERSION}|think={think}")
    return ATTRS / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def extract_attrs(client: Client, src: Path, max_dim: int,
                  max_tokens: int = 700, retries: int = 3) -> tuple[dict, bool, float]:
    """Returns (record, came_from_cache, seconds). The cache flag and timing feed
    the progress display: a fully cached run finishes in under a second and would
    otherwise look like it silently did nothing."""
    ATTRS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cp = attr_cache_path(src, client.model, client.think)
    if cp.exists():
        return json.loads(cp.read_text()), True, time.time() - t0

    small = ensure_small(src, max_dim)
    content = [image_part(small), text_part(ATTR_PROMPT)]

    last = None
    for attempt in range(retries):
        try:
            raw = client.chat(content, max_tokens=max_tokens, temperature=0.0)
            rec = parse_json_blob(raw)
            rec["_file"] = src.name
            # Full path, so stage B can find the source again without assuming
            # every image sits directly in the library root.
            rec["_path"] = str(src.resolve())
            cp.write_text(json.dumps(rec, indent=2))
            return rec, False, time.time() - t0
        except Exception as e:  # noqa: BLE001 - retry on anything transient
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{src.name}: attribute extraction failed - {last}")


# --------------------------------------------------------------------------
# Library categories
# --------------------------------------------------------------------------

# Canonical category -> the garment_type values and folder spellings that mean
# it. Matching is on tokens, so "bras", "bra" and "sports_bras" all resolve.
CATEGORY_TERMS = {
    "bra": {"bra", "bras", "sports_bra", "sports_bras", "bralette", "bralettes",
            "t_shirt_bra", "crop_top", "crop_tops"},
    "legging": {"legging", "leggings", "tight", "tights", "legging_bottoms"},
    "short": {"short", "shorts", "biker_short", "biker_shorts", "cycling_shorts"},
    # Full-length WOVEN bottoms: jeans, chinos, cargos, joggers, trousers. Kept
    # apart from "legging" and "short" for the LAY, which is the only thing a
    # reference is used for. A legging lies as two narrow tubes touching down
    # the middle; a loose jean lies with the legs apart, the rise spread open
    # and the hems well clear of each other. Either one used as the other's lay
    # reference asks the model to reshape the garment, which is the one thing a
    # lay reference must never do.
    #
    # The canonical key is "loose" because that is what the folder is called
    # (library_reference/loose/) - find_categories keys on _canon(folder name),
    # so the key and the folder have to spell it the same way.
    #
    # Terms are disjoint from "short" and "legging" - _canon returns the first
    # set a token is found in, so a word in both would resolve by dict order.
    "loose": {"loose", "loose_fit", "jean", "jeans", "denim", "denims",
              "pant", "pants", "trouser", "trousers", "jogger", "joggers",
              "sweatpant", "sweatpants", "chino", "chinos", "cargo",
              "cargo_pant", "cargo_pants", "wide_leg", "barrel_leg"},
    # The same family as "loose" and split off it for the WAISTBAND: gathered,
    # elasticated, shirred or paperbag, against loose's flat band with belt
    # loops and a fly. That is a lay difference and not just a styling one - a
    # gathered band pulls the whole top of the garment in and stands the frill
    # up off the plate, so the hip sits narrower and the waist sits taller than
    # on a flat-banded jean of the same size.
    #
    # Terms are disjoint from "loose", "legging" and "short" - _canon returns
    # the first set a token is found in, so a word in both would resolve by
    # dict order.
    "boyfriend": {"boyfriend", "boyfriends", "boyfriend_fit", "boyfriend_jean",
                  "boyfriend_jeans", "boyfriend_pant", "boyfriend_pants",
                  "paperbag", "paperbag_pant", "paperbag_pants", "paper_bag",
                  "slouch", "slouchy", "slouch_fit"},
    "top": {"top", "tops", "tank_top", "tank_tops", "tank", "t_shirt", "tee", "tees"},
    # Sleeved upper body, pulled on over the head. Kept apart from "top", which
    # is the sleeveless/tank family: the two are graded on different bands and a
    # tank scored against a sweatshirt library returns confident nonsense.
    "pullover": {"pullover", "pullovers", "sweater", "sweaters", "sweatshirt",
                 "sweatshirts", "jumper", "jumpers", "hoodie", "hoodies",
                 "hooded_sweatshirt", "crewneck", "crewnecks", "crew_neck",
                 "half_zip", "quarter_zip"},
    # Sleeved knit that opens the whole way down the centre front. Split from
    # pullover for the LAY, which is what a reference is for: a cardigan is laid
    # with two front edges meeting on the centre line and the placket flat, and
    # nothing in the pullover library shows that. Used as each other's
    # reference, a pullover asks the model to close a cardigan's front and a
    # cardigan asks it to cut one into a pullover.
    #
    # Terms are disjoint from pullover and fleece - _canon returns the first set
    # a token is found in, so a word in both would resolve by dict order. Note
    # "half_zip"/"quarter_zip" stay with pullover on purpose: those stop above
    # the hem, so the garment still goes over the head.
    "cardigan": {"cardigan", "cardigans", "cardi", "cardis", "button_front",
                 "button_up", "button_through", "open_front", "duster",
                 "shrug", "shrugs", "bolero", "boleros"},
    # Kept apart from "pullover" for the fabric, not the cut. The two share a
    # silhouette, so a fleece scored against the sweatshirt folder returns a
    # confident match on everything except the one attribute that matters: the
    # pile. fabric_finish carries 1.5 of the ~19 total weight, which is not
    # enough to demote a same-cut smooth sweatshirt below a different-cut
    # fleece. Splitting the folder is what makes the pile decisive.
    #
    # Terms are disjoint from "pullover" - _canon returns the first set a token
    # is found in, so a word in both would resolve by dict order.
    "fleece": {"fleece", "fleeces", "fleece_top", "fleece_jacket",
               "fleece_pullover", "polar_fleece", "microfleece", "micro_fleece",
               "sherpa", "sherpas", "sherpa_jacket", "sherpa_pullover"},
}


def _canon(word: str) -> str | None:
    """Map a folder name or a garment_type onto a canonical category, or None."""
    w = re.sub(r"[^a-z]+", "_", str(word).lower()).strip("_")
    for key, terms in CATEGORY_TERMS.items():
        if w in terms or w.rstrip("_s") == key:
            return key
    return None


def find_categories(root: Path) -> dict[str, Path]:
    """Immediate subfolders of root that actually contain images, keyed by
    canonical category (falling back to the literal folder name)."""
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not any(d.glob("*.jpg")):
            continue
        out[_canon(d.name) or d.name.lower()] = d
    return out


def resolve_library(root: Path, garment_type: str,
                    force: str | None = None) -> tuple[list[Path], str]:
    """Pick the images to score against, and explain the choice.

    A flat library is used as-is. With category subfolders the query's own
    garment_type selects one, so a bra query never competes against leggings:
    garment_type carries only 3.0 of the ~19 total weight, so an unrelated
    garment in a similar colour and fabric can still score in the 70s.

    Falls back to pooling every category rather than refusing, so watch mode
    survives an unrecognised garment - but says loudly that it did."""
    cats = find_categories(root)

    if force:
        key = _canon(force) or force.lower()
        if key in cats:
            d = cats[key]
            return sorted(d.glob("*.jpg")), f"{d.name}/ - forced by --category"
        avail = ", ".join(sorted(d.name for d in cats.values())) or "none"
        return [], f"--category {force!r} not found (available: {avail})"

    direct = sorted(root.glob("*.jpg"))
    if direct:
        return direct, f"{root.name}/ - flat layout, {len(direct)} images"
    if not cats:
        return [], f"{root} holds no .jpg files and no category subfolders"

    want = _canon(garment_type)
    if want and want in cats:
        d = cats[want]
        return (sorted(d.glob("*.jpg")),
                f"{d.name}/ - auto-selected from garment_type '{garment_type}'")

    avail = ", ".join(sorted(d.name for d in cats.values()))
    return (sorted(root.rglob("*.jpg")),
            f"POOLED across all categories ({avail}) - garment_type "
            f"'{garment_type}' matched none of them, ranking may be noisy")


# --------------------------------------------------------------------------
# Stage B - scoring
# --------------------------------------------------------------------------

WEIGHTS = {
    "garment_type": 3.0,
    "neckline": 2.5,
    "strap_style": 2.5,
    "strap_width": 1.5,
    "adjusters": 1.5,
    "band": 1.0,
    "fabric_finish": 1.5,
    "padding": 1.0,
    "closure": 1.0,
    "support_level": 1.0,
    "coverage": 1.0,
}

# Partial credit for near-miss values, so "square vs scoop" beats "square vs halter".
PARTIAL = {
    "neckline": {("square", "scoop"): 0.4, ("scoop", "v_neck"): 0.4,
                 ("v_neck", "plunge"): 0.6, ("square", "sweetheart"): 0.3},
    "strap_style": {("wide_straight", "thin_straight"): 0.4,
                    ("crossback", "strappy_multi"): 0.5,
                    ("racerback", "v_back"): 0.4,
                    ("crossback", "v_back"): 0.3},
    "strap_width": {("wide", "medium"): 0.5, ("medium", "thin"): 0.5},
    "support_level": {("high", "medium"): 0.5, ("medium", "low"): 0.5},
    "padding": {("molded_cups", "removable_pads"): 0.4},
    "coverage": {("cropped_short", "standard"): 0.5, ("standard", "longline"): 0.5},
    "band": {("wide_elastic", "logo_band"): 0.5, ("wide_elastic", "narrow_elastic"): 0.5},
    # Both are pile, and "textured" is where a model puts a pile it has not been
    # given a word for - which is what the v2 prompt did to every fleece in the
    # library. Partial credit rather than 0 keeps a fleece near a sherpa and near
    # its own older description, without making either equal to the other.
    "fabric_finish": {("fleece", "sherpa"): 0.5, ("fleece", "textured"): 0.4,
                      ("sherpa", "textured"): 0.3},
}


def field_sim(field: str, a, b) -> float:
    a, b = str(a).lower(), str(b).lower()
    if a == "unknown" or b == "unknown":
        return 0.5  # neither reward nor punish a field the model couldn't see
    if a == b:
        return 1.0
    for (x, y), v in PARTIAL.get(field, {}).items():
        if {a, b} == {x, y}:
            return v
    return 0.0


def hex_to_lab(h: str) -> tuple[float, float, float]:
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex {h!r}")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def color_sim(a: str, b: str) -> float:
    try:
        la, lb = hex_to_lab(a), hex_to_lab(b)
    except Exception:  # noqa: BLE001
        return 0.5
    de = sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5
    return max(0.0, 1.0 - de / 60.0)  # deltaE 60+ counts as fully different


def score(q: dict, c: dict, color_weight: float) -> tuple[float, dict]:
    """Weighted agreement, 0-100.

    A field that is "unknown" on BOTH sides is dropped from the numerator and
    the denominator rather than scoring the usual 0.5. It carries no evidence
    either way, and counting it drags every score toward the middle: the
    schema is bra-shaped, so a leggings-vs-leggings comparison has seven
    inapplicable fields. Scoring those at 0.5 caps two *identical* leggings at
    71.8/100 - under the default 90 pass mark, so a correct match could never
    be reported. Dropping them keeps the scale comparable across categories.
    One-sided "unknown" still scores 0.5: that is real uncertainty."""
    total = 0.0
    max_total = 0.0
    parts = {}
    skipped = []
    for f in FIELDS:
        qv, cv = str(q.get(f, "unknown")).lower(), str(c.get(f, "unknown")).lower()
        if qv == "unknown" and cv == "unknown":
            skipped.append(f)
            continue
        w = WEIGHTS[f]
        s = field_sim(f, qv, cv)
        parts[f] = s
        total += w * s
        max_total += w
    if color_weight > 0:
        s = color_sim(q.get("color_hex", ""), c.get("color_hex", ""))
        parts["color"] = s
        total += color_weight * s
        max_total += color_weight
    if skipped:
        parts["_skipped_both_unknown"] = skipped
    if max_total <= 0:
        return (0.0, parts)   # nothing comparable at all
    return (100.0 * total / max_total, parts)


# --------------------------------------------------------------------------
# Stage C - head-to-head visual pick
# --------------------------------------------------------------------------

# The question here has to be the one the reference actually answers. This
# prompt used to ask for "the same STYLE of garment", and on
# runs/20260827_095158 the model did exactly as asked: it rejected a 99.8 match
# because the query was a multi-colour striped pullover and the candidate a
# solid navy pullover with a teddy bear intarsia - "the pattern and
# construction are fundamentally different, making them distinct styles of
# garments". Every field that bears on the LAY had scored 1.0: pullover, scoop
# neck, ribbed, same band, same coverage.
#
# It was right about style and wrong about the job. select_reference.py
# desaturates the winner to greyscale before installing it, and SKILL.md says
# it twice - a shape and lay reference, never a colour target, never a
# construction reference. Colour and pattern are the two things guaranteed not
# to travel, so rejecting on them vetoes matches the pipeline was built to
# accept, and the only way through was a hand-typed --no-model-veto on every
# run.
#
# The veto is still worth having. It is what catches a candidate that cleared
# the numeric filter on attribute agreement while being the wrong shape to lay
# - the schema cannot see silhouette, so nothing else would. It just has to
# reject on shape rather than on print.
COMPARE_PROMPT = """Image 1 is the QUERY garment (photographed flat, sometimes with a
hangtag - ignore the tag). The following {n} images are candidate reference photos,
labelled {labels}.

The winner is used for ONE purpose: it is converted to greyscale and used as a LAY
reference - a template for how the query garment should be posed and shaped when it
is photographed flat. It is never a colour target and never a construction
reference.

So pick the ONE candidate that the query garment could be laid out to match: same
garment shape and silhouette, same cut. For tops and bras that means neckline shape,
strap or sleeve construction, band and coverage; for bottoms it means rise, leg
length and waistband. Judge only the features the garment actually has - do not look
for a neckline on a pair of leggings.

These do NOT matter and are NOT reasons to reject a candidate:
colour, colourway, stripes, prints, patterns, colour-blocking, logos, embroidery,
intarsia or any applied graphic. A striped garment and a solid one in the same cut
are the SAME LAY. All of it is discarded before the reference is used.

This is a strict match on SHAPE, not a nearest-neighbour choice. The candidates have
already passed a numeric filter, so a pick that cannot be laid the same way is worse
than no pick. If NONE of them shares the query's cut and silhouette, answer "none".
Do not stretch to fill the slot.

Return ONE JSON object, nothing else:
{{"pick": "<label>" or "none",
  "confidence": <0-100 integer, how sure you are the pick works as a lay reference>,
  "runner_up": "<label>" or "none",
  "reason": "<two sentences max, about shape and cut>",
  "differences": "<what still differs in shape between query and your pick, one sentence>"}}"""


def compare_multi(client: Client, query_small: Path, cands: list[tuple[str, Path]]) -> dict:
    labels = [lab for lab, _ in cands]
    content: list = [text_part("Image 1 - QUERY:"), image_part(query_small)]
    for lab, p in cands:
        content.append(text_part(f"Candidate {lab}:"))
        content.append(image_part(p))
    content.append(text_part(COMPARE_PROMPT.format(n=len(cands), labels=", ".join(labels))))
    return parse_json_blob(client.chat(content, max_tokens=500, temperature=0.0))


def compare_composite(client: Client, query_small: Path, cands: list[tuple[str, Path]]) -> dict:
    """Fallback for servers that reject multiple images in one message: paste
    everything into a single labelled contact sheet."""
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    tiles_dir = CACHE / "compare"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)
    entries = [("QUERY", query_small)] + cands
    for i, (lab, p) in enumerate(entries):
        subprocess.run([
            "magick", str(p), "-resize", "460x620", "-gravity", "north",
            "-background", "white", "-splice", "0x44", "-font", font,
            "-pointsize", "34", "-fill", "red", "-annotate", "+0+4", lab,
            str(tiles_dir / f"{i:02d}.jpg"),
        ], check=True, capture_output=True)
    sheet = CACHE / "compare_sheet.jpg"
    subprocess.run(
        ["montage", *sorted(str(p) for p in tiles_dir.glob("*.jpg")),
         "-font", font, "-tile", f"{len(entries)}x", "-geometry", "+6+6",
         "-background", "gray90", str(sheet)],
        check=True, capture_output=True,
    )
    labels = [lab for lab, _ in cands]
    prompt = (
        "This is one contact sheet. The leftmost panel is labelled QUERY. "
        f"The remaining panels are candidates {', '.join(labels)}.\n\n"
        + COMPARE_PROMPT.format(n=len(cands), labels=", ".join(labels))
    )
    return parse_json_blob(
        client.chat([image_part(sheet), text_part(prompt)], max_tokens=500, temperature=0.0)
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def build_result_sheet(query_small: Path, ranked: list[dict], out: Path, k: int,
                       threshold: float = 90.0) -> Path:
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    tiles = CACHE / "result_tiles"
    if tiles.exists():
        shutil.rmtree(tiles)
    tiles.mkdir(parents=True)
    subprocess.run([
        "magick", str(query_small), "-resize", "420x560", "-gravity", "north",
        "-background", "white", "-splice", "0x44", "-font", font,
        "-pointsize", "30", "-fill", "red", "-annotate", "+0+6", "QUERY",
        str(tiles / "00.jpg"),
    ], check=True, capture_output=True)
    for i, r in enumerate(ranked[:k], start=1):
        passed = r["score"] >= threshold
        subprocess.run([
            "magick", str(r["_small"]), "-resize", "420x560", "-gravity", "north",
            "-background", "white", "-splice", "0x44", "-font", font,
            "-pointsize", "30", "-fill", "blue" if passed else "gray60",
            "-annotate", "+0+6",
            f'{r["label"]}  {r["score"]:.0f}' + ("" if passed else "  (no match)"),
            str(tiles / f"{i:02d}.jpg"),
        ], check=True, capture_output=True)
    subprocess.run(
        ["montage", *sorted(str(p) for p in tiles.glob("*.jpg")), "-font", font,
         "-tile", f"{k + 1}x", "-geometry", "+6+6", "-background", "gray90", str(out)],
        check=True, capture_output=True,
    )
    return out


# --------------------------------------------------------------------------
# Watch mode
# --------------------------------------------------------------------------

def stable_sig(p: Path, settle: float = 0.75):
    """(mtime, size) for a file that has stopped changing, else None.

    Reading straight off an mtime bump can catch a half-copied file: Finder and
    `cp` create the entry before the bytes land, and a truncated JPEG makes the
    model hallucinate rather than error. Sampling twice avoids that."""
    try:
        a = (p.stat().st_mtime_ns, p.stat().st_size)
        time.sleep(settle)
        b = (p.stat().st_mtime_ns, p.stat().st_size)
    except FileNotFoundError:
        return None  # mid-replacement; try again next tick
    return a if a == b and a[1] > 0 else None


def watch(client: Client, args) -> int:
    q = Path(args.query)
    print(f"watching {q}")
    print(f"  drop a new image in at that path and it will re-match automatically")
    print(f"  polling every {args.watch_interval}s   -   Ctrl-C to stop")
    print("  will exit on NO MATCH FOUND"
          if not args.keep_watching else
          "  will keep watching through a NO MATCH FOUND")
    print()

    last = None
    runs = 0
    last_run_at = None
    spin = "|/-\\"
    tick = 0
    try:
        while True:
            sig = stable_sig(q)
            if sig is not None and sig != last:
                last = sig
                runs += 1
                stamp = time.strftime("%H:%M:%S")
                kb = sig[1] // 1024
                if IS_TTY:
                    print("\r" + " " * 78)  # clear the idle line
                print("=" * 72)
                print(f"[{stamp}]  run {runs}  -  {q.name} ({kb} KB)")
                print("=" * 72)
                try:
                    rc = do_run(client, args)
                except Exception as e:  # noqa: BLE001 - never let one bad run kill the loop
                    print(f"  run failed: {e}")
                    rc = 1
                last_run_at = time.time()
                print()
                # A no-match is a terminal verdict, not a transient state: stop
                # rather than sit on a query the library cannot satisfy.
                if rc != 0 and not args.keep_watching:
                    why = ("no match found" if rc == 2 else "run failed")
                    print(f"exiting after {runs} run(s): {why}.")
                    print("  (pass --keep-watching to stay running through these)")
                    return rc

            # Idle heartbeat: without it a 24/7 terminal looks frozen and there is
            # no way to tell "waiting" apart from "crashed".
            tick += 1
            if IS_TTY:
                idle = f"{int(time.time() - last_run_at)}s" if last_run_at else "-"
                print(f"\r {spin[tick % 4]} watching {q.name}  |  runs: {runs}  |  "
                      f"idle: {idle}  |  {time.strftime('%H:%M:%S')}  (Ctrl-C to stop)",
                      end="", flush=True)
            time.sleep(args.watch_interval)
    except KeyboardInterrupt:
        print(f"\nstopped after {runs} run(s).")
        return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="default: first model the server reports")
    ap.add_argument("--query", default=str(ROOT / "inputs" / "off_set_image.jpg"))
    ap.add_argument("--library", default=str(ROOT / "library_reference"))
    ap.add_argument("--out-dir", default=str(ROOT),
                    help="where match_results.json and result_top_matches.jpg "
                         "are written (default: the project root)")
    ap.add_argument("--category", default=None,
                    help="force a library subfolder (e.g. bras, leggings) "
                         "instead of picking it from the query's garment_type")
    ap.add_argument("--list-categories", action="store_true",
                    help="show the category subfolders found under --library and exit")
    ap.add_argument("--top-k", type=int, default=5)
    # 0.5, not 2.0. select_reference.py desaturates the winner before installing
    # it, precisely so the reference cannot act as a colour target - so colour
    # must not dominate the choice of which garment to install. At 2.0 it
    # outweighed neckline and strap_style, and a black/cream colour-blocked bra
    # ranked its own style in navy 5th, below three solid-colour V-necks that
    # merely shared a neckline. Stage C then rejected those, correctly, and the
    # run stopped with the right answer sitting unexamined below the threshold.
    # Kept non-zero: colourway is still a weak tiebreak between otherwise
    # identical cuts.
    ap.add_argument("--color-weight", type=float, default=0.5,
                    help="0 = ignore colour, rank on cut/construction only "
                         "(default 0.5: the installed reference is desaturated, "
                         "so colour is a tiebreak, not a driver)")
    ap.add_argument("--threshold", "--min-score", type=float, default=90.0,
                    metavar="SCORE",
                    help="minimum score to count as a match; below it the run "
                         "reports NO MATCH FOUND (default 90, exit code 2)")
    ap.add_argument("--no-model-veto", action="store_true",
                    help="accept the top-scoring candidate even if the model "
                         "answers 'none' in stage C")
    ap.add_argument("--max-dim", type=int, default=1024, help="longest edge sent to the model")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="parallel requests; keep low for a single-GPU server")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--compare-mode", choices=["multi", "composite", "none"], default="multi")
    ap.add_argument("--think", action="store_true",
                    help="let the model reason before answering (~8x slower, "
                         "no measured gain on this task)")
    ap.add_argument("--max-tokens", type=int, default=700,
                    help="answer budget per attribute call (x4 when --think)")
    ap.add_argument("--check", action="store_true",
                    help="probe the server and exit (or fall through to --watch)")
    ap.add_argument("--watch", action="store_true",
                    help="stay running and re-match whenever the query image changes")
    ap.add_argument("--watch-interval", type=float, default=2.0,
                    help="seconds between checks of the query file (default 2)")
    ap.add_argument("--keep-watching", action="store_true",
                    help="in --watch, stay running after a NO MATCH FOUND "
                         "instead of exiting")
    args = ap.parse_args()

    # Python block-buffers stdout when it is not a tty, which in --watch mode
    # makes a redirected log look frozen long after a run has finished.
    sys.stdout.reconfigure(line_buffering=True)

    client = Client(args.base_url, args.model, args.timeout, think=args.think)

    if args.list_categories:
        root = Path(args.library)
        cats = find_categories(root)
        if not cats:
            n = len(sorted(root.glob("*.jpg")))
            print(f"{root}: flat layout, {n} images, no category subfolders")
            return 0
        print(f"{root}:")
        for key, d in sorted(cats.items()):
            print(f"  {d.name:<14} {len(sorted(d.glob('*.jpg'))):>3} images  "
                  f"(matches garment_type: "
                  f"{', '.join(sorted(CATEGORY_TERMS.get(key, {key})))})")
        return 0

    if args.check:
        print(f"probing {args.base_url} ...")
        try:
            ids = client.models()
        except Exception as e:  # noqa: BLE001
            print(f"  UNREACHABLE: {e}")
            print("  -> check the host is up and that this machine has a route to it")
            return 1
        print(f"  reachable. models: {ids}")
        client.model = args.model or ids[0]
        print(f"  smoke-testing vision on {client.model} ...")
        try:
            small = ensure_small(Path(args.query), 512)
            out = client.chat(
                [image_part(small), text_part("In five words or fewer: what garment is this?")],
                max_tokens=40,
            )
            print(f"  vision OK -> {out.strip()!r}")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print("  AUTH FAILED (401): the server needs an API key.")
                print("  -> export QWEN_API_KEY=... , or put the key in .qwen_key")
            else:
                print(f"  VISION FAILED: HTTP {e.code} {e.read().decode()[:200]}")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"  VISION FAILED: {e}")
            print("  -> if the model has no mmproj loaded it is text-only and")
            print("     cannot do this task at all. Check /v1/models capabilities.")
            return 1
        if not args.watch:
            return 0
        print()

    if args.watch:
        return watch(client, args)
    return do_run(client, args)


def do_run(client: Client, args) -> int:
    query = Path(args.query)
    if not query.exists():
        print(f"query not found: {query}", file=sys.stderr)
        return 1

    model = client.resolve_model()
    print(f"model: {model}\n")

    # --- Stage A ---------------------------------------------------------
    # The query is described before the library is even listed: with category
    # subfolders, its garment_type is what decides which folder to search.
    print("stage A: extracting attributes")
    t0 = time.time()
    transient(f"  {'query':>6}  ... asking the model")
    q_attrs, q_cached, q_secs = extract_attrs(client, query, args.max_dim, args.max_tokens)
    settled(f"  {'query':>6}  {'cache' if q_cached else 'MODEL'} {q_secs:5.1f}s  "
            f"{query.name}")
    print(f"          -> {q_attrs.get('garment_type')} / {q_attrs.get('color_name')} / "
          f"{q_attrs.get('neckline')} neck / {q_attrs.get('strap_style')} / "
          f"{q_attrs.get('fabric_finish')}")

    lib, why = resolve_library(Path(args.library),
                              str(q_attrs.get("garment_type", "unknown")),
                              args.category)
    print(f"  library:  {why}")
    if not lib:
        print(f"no images to search under {args.library}", file=sys.stderr)
        return 1
    if len({p.name for p in lib}) != len(lib):
        print("  WARNING: duplicate filenames across categories; results are "
              "keyed on full path, but the printed names will be ambiguous")

    recs: list[dict] = []
    failures: list[str] = []
    n_cached = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(extract_attrs, client, p, args.max_dim, args.max_tokens): p
                for p in lib}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            p = futs[fut]
            try:
                rec, cached, secs = fut.result()
                recs.append(rec)
                n_cached += cached
                # One line per image as it lands, so the terminal shows real work
                # happening rather than a counter that redraws itself and vanishes.
                print(f"  {done:>3}/{len(lib)}  {'cache' if cached else 'MODEL'} "
                      f"{secs:5.1f}s  {p.name[:46]:<46} "
                      f"{rec.get('color_name','?')}, {rec.get('neckline','?')} neck, "
                      f"{rec.get('strap_style','?')}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{p.name}: {e}")
                print(f"  {done:>3}/{len(lib)}  FAIL         {p.name[:46]:<46} {e}")
    sent = len(lib) - n_cached
    print(f"  {'':>3}      {len(lib)} images in {time.time() - t0:.1f}s "
          f"({n_cached} from cache, {sent} sent to the model)"
          + (f", {len(failures)} failed" if failures else ""))

    # --- Stage B ---------------------------------------------------------
    ranked = []
    by_file = {p.name: p for p in lib}          # fallback for pre-_path records
    for r in recs:
        s, parts = score(q_attrs, r, args.color_weight)
        src = Path(r["_path"]) if r.get("_path") else by_file[r["_file"]]
        ranked.append({**r, "score": s, "parts": parts,
                       "_small": ensure_small(src, args.max_dim)})
    ranked.sort(key=lambda r: -r["score"])
    for i, r in enumerate(ranked):
        r["label"] = chr(ord("A") + i) if i < args.top_k else ""

    print(f"\nstage B: ranking (colour weight {args.color_weight}, "
          f"threshold {args.threshold:.0f})")
    print(f"  {'':3} {'score':>6}      {'file':<52} summary")
    for i, r in enumerate(ranked[:max(args.top_k, 10)], start=1):
        summary = (f"{r.get('color_name','?')}, {r.get('neckline','?')} neck, "
                   f"{r.get('strap_style','?')}, {r.get('fabric_finish','?')}")
        flag = "PASS" if r["score"] >= args.threshold else "  - "
        print(f"  {i:>2}. {r['score']:6.1f} {flag}  {r['_file']:<52} {summary}")

    # A candidate below the threshold is not a match, so it is not eligible to
    # win stage C either. Gate before spending the comparison call.
    qualifying = [r for r in ranked if r["score"] >= args.threshold][:args.top_k]
    top = ranked[:args.top_k]
    best = ranked[0]

    # --- Stage C ---------------------------------------------------------
    verdict = None
    if not qualifying:
        print(f"\nNO MATCH FOUND - nothing reached {args.threshold:.0f}.")
        print(f"  closest was {best['_file']} at {best['score']:.1f} "
              f"({args.threshold - best['score']:.1f} short)")
        print(f"  the library has no {q_attrs.get('neckline','?')}-neck / "
              f"{q_attrs.get('strap_style','?')} / {q_attrs.get('fabric_finish','?')} "
              f"piece close enough to this query")
        print("  skipping stage C; see result_top_matches.jpg for the near misses")
    elif args.compare_mode != "none":
        # Run even for a single survivor: under a strict threshold the useful
        # question is "is this actually a match", not just "which is closest".
        what = ("confirming the 1 candidate that cleared" if len(qualifying) == 1
                else f"visual head-to-head on the {len(qualifying)} candidates that cleared")
        print(f"\nstage C: {what} {args.threshold:.0f}")
        cands = [(r["label"], r["_small"]) for r in qualifying]
        q_small = ensure_small(query, args.max_dim)
        tC = time.time()
        transient(f"  sending {len(cands) + 1} images, waiting for the model ...")
        try:
            if args.compare_mode == "multi":
                verdict = compare_multi(client, q_small, cands)
            else:
                verdict = compare_composite(client, q_small, cands)
        except Exception as e:  # noqa: BLE001
            print(f"  {args.compare_mode} mode failed ({e}); retrying as composite")
            try:
                verdict = compare_composite(client, q_small, cands)
            except Exception as e2:  # noqa: BLE001
                print(f"  composite also failed: {e2}")
        settled(f"  model replied in {time.time() - tC:.1f}s")
        if verdict:
            by_label = {r["label"]: r for r in qualifying}
            raw = str(verdict.get("pick", "")).strip().upper()
            pick = by_label.get(raw)
            conf = verdict.get("confidence")
            if raw in ("NONE", "", "NULL"):
                print(f"  pick:        none - the model rejected all "
                      f"{len(qualifying)} candidate(s)")
            else:
                print(f"  pick:        {verdict.get('pick')}"
                      + (f"  -> {pick['_file']}" if pick
                         else "  (label not among the qualifying candidates)"))
            if conf is not None:
                print(f"  confidence:  {conf}")
            print(f"  runner-up:   {verdict.get('runner_up')}")
            print(f"  reason:      {verdict.get('reason')}")
            print(f"  differences: {verdict.get('differences')}")

    # --- Decision ---------------------------------------------------------
    # Two independent gates, both must pass: the numeric score clears the
    # threshold, and the model does not reject the survivors on sight.
    match_found = bool(qualifying)
    vetoed = False
    chosen = None
    if match_found:
        chosen = qualifying[0]
        if verdict:
            raw = str(verdict.get("pick", "")).strip().upper()
            by_label = {r["label"]: r for r in qualifying}
            if raw in ("NONE", "", "NULL"):
                vetoed = True
            elif raw in by_label:
                chosen = by_label[raw]
        if vetoed and not args.no_model_veto:
            match_found = False
            chosen = None
            print(f"\nNO MATCH FOUND - {best['score']:.1f} cleared the "
                  f"{args.threshold:.0f} threshold, but the model rejected the "
                  f"candidate(s) on inspection.")
            print("  -> rerun with --no-model-veto to trust the score alone")
        elif vetoed:
            print(f"\n  model said none, but --no-model-veto is set; "
                  f"keeping the top-scoring candidate")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = build_result_sheet(ensure_small(query, args.max_dim), ranked,
                               out_dir / "result_top_matches.jpg", args.top_k,
                               args.threshold)
    out = {
        "model": model,
        "query": str(query),
        "query_attrs": q_attrs,
        "library_root": str(args.library),
        "library_used": why,
        "library_count": len(lib),
        "color_weight": args.color_weight,
        "threshold": args.threshold,
        "match_found": match_found,
        "match": chosen["_file"] if chosen else None,
        # The full path as well as the basename: the caller has to copy this
        # file, and a basename does not say which category folder it came from.
        "match_path": chosen.get("_path") if chosen else None,
        "match_score": round(chosen["score"], 1) if chosen else None,
        "model_confidence": (verdict or {}).get("confidence"),
        "model_vetoed": vetoed,
        "n_qualifying": len(qualifying),
        "ranked": [{k: v for k, v in r.items() if not k.startswith("_small")}
                   for r in ranked],
        "verdict": verdict,
        "failures": failures,
    }
    (out_dir / "match_results.json").write_text(
        json.dumps(out, indent=2, default=str))

    print()
    if match_found:
        print(f"MATCH: {chosen['_file']}  ({chosen['score']:.1f}/100)")
    elif vetoed:
        print(f"NO MATCH FOUND  (model rejected all {len(qualifying)} candidate(s) "
              f"that cleared {args.threshold:.0f})")
    else:
        print(f"NO MATCH FOUND  (best {best['score']:.1f} < {args.threshold:.0f})")
    print(f"wrote {out_dir / 'match_results.json'}")
    print(f"wrote {sheet}")
    return 0 if match_found else 2


if __name__ == "__main__":
    sys.exit(main())
