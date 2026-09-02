#!/usr/bin/env python3
"""Step 0 - choose the lay reference from the library, before the agent runs.

    python tools/select_reference.py --run runs/20260901_140159
    python tools/select_reference.py --index            # describe the library, exit
    python tools/select_reference.py --calibrate        # where to put --threshold
    python tools/select_reference.py --dry-run          # decide, install nothing

Picking the reference used to be a human step: look at the garment photo, find
the closest laydown in the library, desaturate it, drop it in inputs/. This does
the same thing the same way every time.

Three stages, in decreasing order of how much they can be trusted:

  A. DESCRIBE.  One vision call per image fills in a fixed form. Cached on disk
     and keyed on content, so a library is described once and every run after
     that is free. The query is described too, by the same call and the same
     form - the comparison in stage B is only meaningful if both sides were
     asked identical questions.

  B. SCORE.     Plain arithmetic over those forms, no model. Disqualify what
     cannot work at all, then weight what is left into a number out of 100.
     Free, repeatable, and it cannot hallucinate a match.

  C. CONFIRM.   One vision call showing the query and the survivors side by
     side. It can only ever REJECT - it is a veto on B, not a replacement for
     it. Both gates must pass.

Two things it insists on, both because of how the rest of the harness behaves:

  * GREYSCALE. The reference is a shape and lay reference, never a colour
    target. Sent in colour it is read as one, and the garment comes back wearing
    the reference's tone. --colour opts out.
  * TONE-MATCHED. Greyscale strips hue and keeps lightness, and lightness is
    copied just as readily: runs/20260901_222258 put an L* 80 reference in
    front of an L* 45 garment and every candidate that copied the pose shipped
    pale, dE 37 from the product. So the reference garment is scaled in linear
    light until its mean lightness equals the source garment's, plate untouched
    - common.prepare_reference_image(). --no-tone-match opts out.
  * ONE answer, written down either way. reference_selection.json is produced
    for BOTH outcomes, carrying `match_found`, so a caller that routes on the
    answer has one file that always exists and always answers the same question.
    A receipt that appeared only on success would be indistinguishable from a
    run that died before step 0.

Exit codes:  0 installed   2 no match, nothing installed   1 broke

2 is a business outcome, not a fault: the library holds nothing close enough to
this garment, and a human has to supply a reference.

Rebuilt from old/tools/{match_reference,select_reference}.py. The mechanics are
the same - cached descriptions, weighted scoring, both-blank fields dropped, a
model veto, a construction-bleed scan. What changed is that the form is now
garment-neutral and the library is flat, so nothing here has to be taught what a
new kind of garment is called.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image

import common as C

Image.MAX_IMAGE_PIXELS = None

SMALL = C.REFCACHE / "small"
ATTRS = C.REFCACHE / "attrs"
TONE = C.REFCACHE / "tone"

# The one path the harness reads. prepare_reference() writes the same file on
# the operator-supplied path, so whichever way the reference arrived, image 2 of
# turn 1 is this file. The library asset itself is never written to - it is read,
# converted, and the result lands here; the untouched colour original is copied
# in beside it as common.REFERENCE_ORIGINAL.
CANON = C.REFERENCE_GREY

IS_TTY = sys.stdout.isatty()


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

# Bump when ATTR_PROMPT or the field list changes, to invalidate cached records.
# A cached record answers the question it was ASKED, and a form that has gained
# a field or changed a rule is a different question - merging the two silently
# scores an old answer against a new one.
#
# v1: first garment-neutral form. Replaces the bra-shaped schema in
# old/tools/match_reference.py (v8), which asked for strap style, strap width,
# adjusters, padding, band and support level. Those are blank on anything that
# is not a bra, and the old code's answer was a growing pile of "if the garment
# is a BOTTOM, set these seven to unknown" special cases plus a hand-maintained
# list of which folder each garment word belonged in. Every new garment family
# cost a prompt revision. This form asks only things any garment can answer.
PROMPT_VERSION = "gn1"

ATTR_PROMPT = """You are cataloguing flat-lay product photographs of garments.

Look at the garment in this image and fill in the form below. Judge only what you
can actually see. If a field is genuinely not visible or does not apply to this
garment, answer "unknown" - a field answered "unknown" on both sides of a
comparison is dropped from scoring, so it costs you nothing to be honest.

Ignore any hangtag, price ticket, label, pin, clip, background, shadow or prop.
Describe the garment only.

Judge the garment's CONSTRUCTION, not how it happens to be arranged in this
photograph. A sleeve folded across the body is still a long sleeve. A leg
crumpled short is still a full-length leg.

Two fields decide most of the answer, so take them slowly:

BODY_REGION. "upper" for anything worn above the waist - bra, top, shirt,
sweater, jacket. "lower" for anything worn below it - leggings, jeans, shorts,
skirt. "full_body" for a single garment covering both - dress, jumpsuit,
all-in-one.

FRONT_OPENING. Look at the centre front and decide how far it separates.
"full_open" means two separate front panels meeting down the middle, held by
buttons, snaps or a zip running all the way from neck to hem, or hanging open -
a cardigan, a jacket. "partial_placket" means a placket, half-zip or
quarter-zip that stops somewhere above the hem, so the garment still has to go
over the head. "closed" means one unbroken front panel. "wrap" means one panel
crosses over the other and ties.

Return ONE JSON object, nothing else. No prose, no markdown fence.

{
  "garment_type":    "<plain english, one to three words, e.g. hooded sweatshirt, wide-leg jean, sports bra>",
  "body_region":     "upper" | "lower" | "full_body" | "unknown",
  "silhouette":      "fitted" | "straight" | "relaxed" | "oversized" | "a_line" | "flared" | "unknown",
  "length":          "cropped" | "hip" | "thigh" | "knee" | "midi" | "ankle" | "full" | "unknown",
  "sleeve":          "none" | "cap" | "short" | "elbow" | "three_quarter" | "long" | "unknown",
  "leg":             "none" | "short" | "knee" | "cropped" | "full" | "unknown",
  "front_opening":   "closed" | "partial_placket" | "full_open" | "wrap" | "unknown",
  "neckline":        "crew" | "v" | "scoop" | "square" | "high" | "hooded" | "collared" | "plunge" | "halter" | "strapless" | "none" | "unknown",
  "waist_treatment": "flat_band" | "elastic" | "gathered" | "drawstring" | "none" | "unknown",
  "hem":             "straight" | "curved" | "ribbed" | "raw" | "cuffed" | "unknown",
  "fabric_finish":   "smooth" | "ribbed" | "textured" | "pile" | "chunky_knit" | "lace" | "mesh" | "denim" | "printed" | "unknown",
  "structure":       "soft_drape" | "medium" | "structured" | "unknown",
  "color_name":      "<plain english, e.g. periwinkle blue>",
  "color_hex":       "#RRGGBB",
  "notes":           "<one short sentence on the single most distinctive feature>"
}

Notes on the trickier fields:

  "sleeve" is "none" for a sleeveless garment - a tank, a bra, a vest - and
  "unknown" for a bottom, which has no sleeves to have a length.
  "leg" is "none" for a skirt, and "unknown" for anything worn on the top half.
  "length" is how far down the body the garment reaches, judged against the
  wearer, not the photograph: "cropped" above the natural waist, "hip" at the
  hip, then thigh, knee, midi, ankle, "full" to the floor or the foot.
  "structure" is whether the garment holds a shape on its own: "soft_drape" for
  jersey and fine knit that pools, "structured" for denim, canvas, a moulded cup
  or anything padded, "medium" in between.
"""

# Weights sum to 22.5 - the same order of magnitude the old form used, so a
# score out of 100 means roughly what it used to. The four heaviest are the four
# that decide whether one garment's laydown can be copied onto another at all;
# hem and structure are detail, and colour is a tiebreak between otherwise
# identical cuts because the winner is desaturated before it is ever used.
WEIGHTS = {
    "garment_type": 3.0,
    "silhouette": 2.5,
    "length": 2.5,
    "sleeve": 2.5,
    "leg": 2.5,
    "front_opening": 2.0,
    "neckline": 2.0,
    "waist_treatment": 1.5,
    "fabric_finish": 1.5,
    "hem": 1.0,
    "structure": 1.0,
}
COLOUR_WEIGHT_DEFAULT = 0.5

# Tone: the MEASURED lightness gap between the two garments, in L*, read off
# the pixels rather than off the form. Not a field and not part of the
# threshold. The threshold was calibrated on construction alone, and the
# reference is re-toned before it is used, so tone can neither make a
# candidate a match nor stop it being one. What it can do is order the
# candidates that ARE matches: a reference already close in tone needs a
# smaller correction and keeps more of its own shading. So it is a penalty on
# the RANK, in score points: 0 for an equal tone, the full TONE_WEIGHT at
# TONE_SPAN_L apart or more. 4 points is what a 1.0-weight field is worth on
# this scale - detail, below every construction field.
TONE_WEIGHT_DEFAULT = 4.0
TONE_SPAN_L = 40.0
# Bump when common.garment_tone() changes what it measures.
# t2: the statistic moved from the whole-mask mean to the fully lit fabric
# (common.TONE_LIT_PCT), so every cached tone is a different number.
TONE_VERSION = "t2"

FIELDS = list(WEIGHTS)

# PROVISIONAL, and deliberately not the old pipeline's 95. That number was
# measured against a bra-shaped form with different fields, different weights
# and more of them agreeing exactly. It does not transfer.
#
# Where 78 comes from: --calibrate over a 9-image spread of this project's own
# assets - three cardigans, a sweater, a hooded sweatshirt, leggings, a jean,
# leather trousers and a sports bra. 29 of the 36 pairs were disqualified
# outright, and the 7 that were scored split cleanly:
#
#     same garment type        n=3   79.5  81.3  90.7
#     different garment type   n=4   65.2 .. 73.2
#
# 78 sits in that gap. It is the widest evidence available and it is still only
# nine images: a real library holds more near-duplicates, which raises the top
# of the same-type range, and more near-misses, which raises the top of the
# other one. --calibrate reads the real number off whatever library is actually
# installed, and both this tool and the harness say the default is provisional
# until someone has run it.
DEFAULT_THRESHOLD = 78.0

# Ordered scales. Adjacent values score 0.5, two apart 0.25, further apart 0.
# Generated rather than written out as pairs: the old form's PARTIAL table was
# 20 hand-written tuples and a value added to an enum silently scored 0 against
# its own neighbour until someone remembered to extend it.
ORDERED = {
    "sleeve": ["none", "cap", "short", "elbow", "three_quarter", "long"],
    "leg": ["none", "short", "knee", "cropped", "full"],
    "length": ["cropped", "hip", "thigh", "knee", "midi", "ankle", "full"],
    "silhouette": ["fitted", "straight", "relaxed", "oversized"],
    "structure": ["soft_drape", "medium", "structured"],
    "waist_treatment": ["flat_band", "elastic", "drawstring", "gathered"],
}

# Unordered near-misses, where one value is a plausible reading of the other.
NEAR = {
    # "hooded" is deliberately far from everything: a hood is a whole extra
    # piece of garment lying above the shoulders, and it is the single most
    # visible thing about a laydown that has one. Everything else here is a
    # difference in where an edge sits.
    "neckline": {("crew", "scoop"): 0.5, ("scoop", "v"): 0.4,
                 ("v", "plunge"): 0.6, ("crew", "square"): 0.4,
                 ("high", "crew"): 0.5, ("crew", "collared"): 0.4,
                 ("high", "collared"): 0.5, ("v", "collared"): 0.3,
                 ("hooded", "collared"): 0.2, ("hooded", "high"): 0.2,
                 ("strapless", "halter"): 0.2},
    # "textured" is the hub, because it is where the model puts a surface it has
    # not been given a better word for - on this library it called one chunky
    # cardigan "chunky_knit" and a near-identical one "textured", which scored
    # 0.0 on a field worth 1.5. The old pipeline learned the same thing about
    # fleece and pile. So every non-smooth finish keeps partial credit against
    # "textured", without any two of them becoming equal to each other.
    "fabric_finish": {("textured", "ribbed"): 0.5, ("textured", "pile"): 0.5,
                      ("textured", "chunky_knit"): 0.5,
                      ("textured", "denim"): 0.4, ("textured", "printed"): 0.4,
                      ("textured", "mesh"): 0.3, ("textured", "lace"): 0.3,
                      ("textured", "smooth"): 0.3,
                      ("pile", "chunky_knit"): 0.4,
                      ("chunky_knit", "ribbed"): 0.4,
                      ("smooth", "printed"): 0.4, ("denim", "smooth"): 0.3,
                      ("mesh", "lace"): 0.4},
    "hem": {("straight", "curved"): 0.5, ("ribbed", "cuffed"): 0.6,
            ("straight", "raw"): 0.5, ("straight", "ribbed"): 0.3},
    "front_opening": {("closed", "partial_placket"): 0.5,
                      ("full_open", "wrap"): 0.5},
    "silhouette": {("a_line", "flared"): 0.6, ("relaxed", "a_line"): 0.3,
                   ("straight", "a_line"): 0.3, ("flared", "oversized"): 0.2},
}

# Values that mean "this garment does not have that part", as opposed to "it has
# one and I could not see it". The difference matters to the disqualifiers: a
# sleeveless top and a long-sleeved one cannot share a laydown, but a sleeve
# somebody could not make out is just missing evidence.
ABSENT = "none"


def disqualify(q: dict, c: dict) -> str | None:
    """Why `c` cannot be `q`'s lay reference at all, or None if it can be.

    Excluded, not scored low. The old pipeline kept the library in per-garment
    folders and only ever scored a bra against other bras; a flat library gives
    that protection up, and scoring a mismatch to a low number does not replace
    it - garment_type is 3.0 of 22.5, so a jean and a legging that agree on
    everything else still reach the 80s, and the threshold is the only thing
    standing between that and a reference that asks the generator to reshape the
    garment.

    Each of these is a difference that makes the LAY untransferable, and each is
    a fact about the garment rather than about a category anyone has to name:

      * a top's laydown says nothing about how a pair of jeans lies
      * a sleeveless garment has no sleeves to put where the reference's are
      * a skirt has no legs to arrange like the reference's two
      * a cardigan lies with two front panels meeting on the centre line, and a
        pullover has nowhere to put them

    'unknown' never disqualifies. It is missing evidence, and refusing on it
    would throw away the whole library every time the model hedged.
    """
    def v(rec, f):
        return str(rec.get(f, "unknown")).lower()

    qr, cr = v(q, "body_region"), v(c, "body_region")
    if qr != "unknown" and cr != "unknown" and qr != cr:
        return f"body_region {cr} vs {qr}"

    for part in ("sleeve", "leg"):
        qp, cp = v(q, part), v(c, part)
        if "unknown" in (qp, cp):
            continue
        if (qp == ABSENT) != (cp == ABSENT):
            return (f"{part}: one has none and the other has "
                    f"{cp if qp == ABSENT else qp}")

    qf, cf = v(q, "front_opening"), v(c, "front_opening")
    if "unknown" not in (qf, cf):
        opens = {"full_open"}
        if (qf in opens) != (cf in opens):
            return f"front_opening {cf} vs {qf}"
    return None


def field_sim(field: str, a: str, b: str) -> float:
    """Agreement on one field, 0-1."""
    a, b = str(a).lower(), str(b).lower()
    if a == "unknown" or b == "unknown":
        return 0.5          # neither reward nor punish what could not be seen
    if a == b:
        return 1.0
    scale = ORDERED.get(field)
    if scale and a in scale and b in scale:
        return {1: 0.5, 2: 0.25}.get(abs(scale.index(a) - scale.index(b)), 0.0)
    for (x, y), val in NEAR.get(field, {}).items():
        if {a, b} == {x, y}:
            return val
    return 0.0


def _words(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", str(s).lower()) if len(w) > 2}


def _head(s: str) -> str:
    """The head noun of a garment name - the last word that is not a modifier.

    English compounds put the head last: a 'hooded sweatshirt' is a sweatshirt,
    a 'wide-leg jean' is a jean, a 'knit cardigan' is a cardigan. Plurals are
    folded in so 'jean' and 'jeans' are the same word.
    """
    ws = [w for w in re.split(r"[^a-z]+", str(s).lower()) if w]
    if not ws:
        return ""
    w = ws[-1]
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def type_sim(a: str, b: str) -> float:
    """garment_type is free text, so it is compared as words, not as an enum.

    Scored on the HEAD NOUN, not on the whole phrase. This is the fix for the
    single largest source of lost points on a correct match: asked for a garment
    name in plain english, the model answers 'knit cardigan', 'hooded cardigan'
    and 'cardigan' for three cardigans, and comparing phrases scored all three
    pairs at 0.5 - 1.5 of 22.5 given away on every genuine match, about 7 points
    of the final score, measured on this library.

    The head noun is what the garment IS; everything before it is a modifier
    that the other fields already measure - 'hooded' is the neckline, 'knit' is
    the fabric, 'wide-leg' is the silhouette. Counting them here charges the
    same difference twice.

    Full credit for the same head. A shared modifier with a different head is
    worth little: 'hooded sweatshirt' and 'hooded cardigan' have a word in
    common and are not the same garment.
    """
    if not a or not b or "unknown" in (str(a).lower(), str(b).lower()):
        return 0.5
    ha, hb = _head(a), _head(b)
    if not ha or not hb:
        return 0.5
    if ha == hb:
        return 1.0
    wa, wb = _words(a), _words(b)
    return 0.25 if wa & wb else 0.0


def hex_to_rgb(h: str):
    h = (h or "").strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex {h!r}")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def colour_sim(a: str, b: str) -> float:
    """Agreement in HUE and saturation only: the a*b* distance between the two
    reported hexes, 1.0 for the same colour, 0 at 60 apart. Lightness is left
    out on purpose - it is measured off the pixels and handled as tone, see
    tone_sim() - and it was what this term used to be wrong about: it scored
    two greys 35 L* apart as different colours, at a weight too small to
    matter, and that gap was the one that bled."""
    try:
        import numpy as np
        la = C.srgb_to_lab(np.array(hex_to_rgb(a), dtype=float))
        lb = C.srgb_to_lab(np.array(hex_to_rgb(b), dtype=float))
        de = float(np.linalg.norm(la[1:] - lb[1:]))
    except Exception:  # noqa: BLE001 - a missing or malformed hex is not fatal
        return 0.5
    return max(0.0, 1.0 - de / 60.0)


def score(q: dict, c: dict, colour_weight: float) -> tuple[float, dict]:
    """Weighted agreement, 0-100, plus the per-field breakdown.

    A field that is "unknown" on BOTH sides is dropped from the numerator AND
    the denominator rather than scoring the usual 0.5. It carries no evidence
    either way, and counting it drags every score toward the middle: a pair of
    jeans has no sleeve, no neckline and no bra fields to fill in, and scoring
    those at 0.5 caps two IDENTICAL jeans below the pass mark, so a correct
    match could never be reported. This is the single most important line in the
    file and it was paid for in the old pipeline, where two identical leggings
    capped at 71.8/100 against a threshold of 90.

    One-sided "unknown" still scores 0.5: that is real uncertainty.
    """
    total = max_total = 0.0
    parts: dict = {}
    skipped: list[str] = []
    for f in FIELDS:
        qv = str(q.get(f, "unknown")).lower()
        cv = str(c.get(f, "unknown")).lower()
        if qv == "unknown" and cv == "unknown":
            skipped.append(f)
            continue
        s = type_sim(qv, cv) if f == "garment_type" else field_sim(f, qv, cv)
        parts[f] = round(s, 3)
        total += WEIGHTS[f] * s
        max_total += WEIGHTS[f]
    if colour_weight > 0:
        s = colour_sim(q.get("color_hex", ""), c.get("color_hex", ""))
        parts["hue"] = round(s, 3)
        total += colour_weight * s
        max_total += colour_weight
    if skipped:
        parts["_skipped_both_unknown"] = skipped
    if max_total <= 0:
        return 0.0, parts
    return 100.0 * total / max_total, parts


def tone_sim(dL: float | None) -> float:
    """1.0 for an equal tone, falling to 0 at TONE_SPAN_L apart. None - one
    side could not be measured - is 0.5, real uncertainty, as for a one-sided
    unknown field."""
    if dL is None:
        return 0.5
    return max(0.0, 1.0 - abs(dL) / TONE_SPAN_L)


def rank_score(score: float, dL: float | None, tone_weight: float) -> float:
    """The order candidates are shown in and chosen from. The threshold is
    judged on `score`; this can only move a candidate DOWN the list, never
    across the line."""
    return score - tone_weight * (1.0 - tone_sim(dL))


def tone_cache_path(img: Path) -> Path:
    key = f"{img.resolve()}|{img.stat().st_mtime_ns}|{TONE_VERSION}"
    return TONE / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def tone_of(img: Path) -> dict:
    """common.garment_tone(), cached on path and mtime like the descriptions.
    It is a second of numpy per image, and a library is measured once."""
    TONE.mkdir(parents=True, exist_ok=True)
    cp = tone_cache_path(img)
    if cp.exists():
        try:
            return json.loads(cp.read_text())
        except (json.JSONDecodeError, OSError):
            cp.unlink(missing_ok=True)
    t = C.garment_tone(img)
    cp.write_text(json.dumps(t))
    return t


def fmt_dL(dL: float | None) -> str:
    return "  n/a " if dL is None else f"{dL:+5.1f}"


# ---------------------------------------------------------------------------
# The vision server
# ---------------------------------------------------------------------------

def _base_url() -> str:
    """The harness's own server, with /v1 on the end however it was configured.

    Deliberately NOT a separate REFMATCH_BASE_URL, which is what the old
    pipeline used. There was a second vision box then; there is one proxy now,
    and two variables for one endpoint meant one of them was always stale.
    """
    url = C.conf("QWEN_BASE_URL", "http://10.11.245.41:8091").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class Client:
    def __init__(self, base_url: str = "", model: str = "", timeout: int = 300):
        self.base_url = (base_url or _base_url()).rstrip("/")
        self.model = model or C.conf("QWEN_MODEL", "")
        self.key = C.conf("QWEN_API_KEY", "not-needed")
        self.timeout = timeout

    def _post(self, path: str, payload: dict | None = None, timeout: int = 0):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
            method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        ids = [m["id"] for m in self._post("/models", timeout=20).get("data", [])]
        if not ids:
            raise RuntimeError(f"{self.base_url} serves no models")
        self.model = ids[0]
        return self.model

    def chat(self, content: list, max_tokens: int = 900) -> str:
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0.0,
                   # The harness's model is a reasoning model. With thinking on,
                   # the answer lands in `reasoning_content` and `content` comes
                   # back empty until the chain finishes - measured ~8x slower
                   # here for no gain on a form-filling task.
                   "chat_template_kwargs": {"enable_thinking": False}}
        out = self._post("/chat/completions", payload)
        choice = out["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text and choice.get("finish_reason") == "length":
            raise RuntimeError("model spent its whole budget reasoning and "
                               "returned no answer; raise --max-tokens")
        return text


def data_url(path: Path) -> str:
    return ("data:image/jpeg;base64,"
            + base64.b64encode(Path(path).read_bytes()).decode())


def image_part(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def text_part(s: str) -> dict:
    return {"type": "text", "text": s}


def parse_json_blob(text: str) -> dict:
    """Models wrap JSON in fences or prose often enough that this is worth it."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Stage A - describe
# ---------------------------------------------------------------------------

def attr_cache_path(img: Path, model: str) -> Path:
    key = f"{img.resolve()}|{img.stat().st_mtime_ns}|{model}|{PROMPT_VERSION}"
    return ATTRS / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def describe(client: Client, src: Path, max_dim: int = 1024,
             max_tokens: int = 700, retries: int = 3) -> tuple[dict, bool, float]:
    """Fill in the form for one image. Returns (record, from_cache, seconds).

    The cache flag and the timing feed the progress display: a fully cached run
    finishes in under a second and would otherwise look like it silently did
    nothing.
    """
    ATTRS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cp = attr_cache_path(src, client.model)
    if cp.exists():
        try:
            return json.loads(cp.read_text()), True, time.time() - t0
        except (json.JSONDecodeError, OSError):
            cp.unlink(missing_ok=True)      # a truncated write, not a verdict

    content = [image_part(C.ensure_small(src, max_dim, SMALL)),
               text_part(ATTR_PROMPT)]
    last = None
    for attempt in range(retries):
        try:
            rec = parse_json_blob(client.chat(content, max_tokens=max_tokens))
            rec["_file"] = src.name
            rec["_path"] = str(src.resolve())
            cp.write_text(json.dumps(rec, indent=2))
            return rec, False, time.time() - t0
        except Exception as e:  # noqa: BLE001 - retry anything transient
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{src.name}: could not be described - {last}")


def library_images(root: Path) -> list[Path]:
    """Every image in the library. Flat, but rglob'd anyway so that a library
    somebody has organised into folders still works - the folders carry no
    meaning here, the disqualifiers do that job."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts
                  and not p.name.startswith("."))


def describe_all(client: Client, paths: list[Path], concurrency: int,
                 max_dim: int, max_tokens: int,
                 quiet: bool = False) -> tuple[list[dict], list[str], int]:
    """Describe every path, in parallel. Returns (records, failures, n_cached)."""
    recs: list[dict] = []
    failures: list[str] = []
    n_cached = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(describe, client, p, max_dim, max_tokens): p
                for p in paths}
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            p = futs[fut]
            try:
                rec, cached, secs = fut.result()
                recs.append(rec)
                n_cached += cached
                if not quiet:
                    # One line per image as it lands, so the terminal shows real
                    # work happening rather than a counter that redraws itself.
                    print(f"  {done:>3}/{len(paths)}  "
                          f"{'cache' if cached else 'MODEL'} {secs:5.1f}s  "
                          f"{p.name[:44]:<44} {rec.get('garment_type', '?')}, "
                          f"{rec.get('color_name', '?')}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{p.name}: {e}")
                print(f"  {done:>3}/{len(paths)}  FAIL         "
                      f"{p.name[:44]:<44} {e}")
    return recs, failures, n_cached


# ---------------------------------------------------------------------------
# Stage C - the veto
# ---------------------------------------------------------------------------

# The question here has to be the one the reference actually answers. The old
# pipeline's version asked for "the same STYLE of garment", and the model did
# exactly as asked: it rejected a 99.8 match because the query was a striped
# pullover and the candidate a solid navy one with a teddy bear intarsia -
# "fundamentally different styles". Every field bearing on the LAY had scored
# 1.0. It was right about style and wrong about the job, because the winner is
# desaturated before it is used and colour and pattern are the two things
# guaranteed not to travel.
#
# The veto is still worth having: it is what catches a candidate that agreed on
# every field while being the wrong SHAPE to lay, which no form can see. It just
# has to reject on shape rather than on print.
COMPARE_PROMPT = """Image 1 is the QUERY garment, photographed flat, sometimes with a
hangtag - ignore the tag. The following {n} images are candidate reference
photographs, labelled {labels}.

The winner is used for ONE purpose: it is converted to greyscale and used as a
LAY reference - a template for how the query garment should be posed and shaped
when it is photographed flat. It is never a colour target and never a
construction reference.

So pick the ONE candidate that the query garment could be LAID OUT to match:
same overall shape and silhouette, same cut, the same parts in the same places.
Judge only the parts the garment actually has - do not look for a neckline on a
pair of trousers, or a leg on a jacket.

These do NOT matter and are NOT reasons to reject a candidate:
colour, colourway, stripes, prints, patterns, colour-blocking, logos,
embroidery, intarsia, appliqué or any applied graphic. A striped garment and a
solid one in the same cut are the SAME LAY. All of it is discarded before the
reference is used.

This is a strict match on SHAPE, not a nearest-neighbour choice. The candidates
have already passed a numeric filter, so a pick that cannot be laid the same way
is worse than no pick at all. If NONE of them shares the query's cut and
silhouette, answer "none". Do not stretch to fill the slot.

Return ONE JSON object, nothing else:
{{"pick": "<label>" or "none",
  "confidence": <0-100 integer, how sure you are the pick works as a lay reference>,
  "runner_up": "<label>" or "none",
  "reason": "<two sentences max, about shape and cut>",
  "differences": "<what still differs in shape between query and your pick, one sentence>"}}"""


def compare_multi(client: Client, query_small: Path,
                  cands: list[tuple[str, Path]]) -> dict:
    labels = [lab for lab, _ in cands]
    content: list = [text_part("Image 1 - QUERY:"), image_part(query_small)]
    for lab, p in cands:
        content += [text_part(f"Candidate {lab}:"), image_part(p)]
    content.append(text_part(COMPARE_PROMPT.format(n=len(cands),
                                                   labels=", ".join(labels))))
    return parse_json_blob(client.chat(content, max_tokens=500))


def compare_composite(client: Client, query_small: Path,
                      cands: list[tuple[str, Path]]) -> dict:
    """Fallback for a server that rejects several images in one message: paste
    everything into a single labelled strip and ask about that instead."""
    sheet = C.contact_sheet([("QUERY", query_small)] + list(cands),
                            C.REFCACHE / "compare_sheet.jpg")
    labels = ", ".join(lab for lab, _ in cands)
    prompt = (f"This is one contact sheet. The leftmost panel is labelled "
              f"QUERY. The remaining panels are candidates {labels}.\n\n"
              + COMPARE_PROMPT.format(n=len(cands), labels=labels))
    return parse_json_blob(
        client.chat([image_part(sheet), text_part(prompt)], max_tokens=500))


# ---------------------------------------------------------------------------
# Construction bleed
# ---------------------------------------------------------------------------

# Words that name how a garment is BUILT rather than how it is laid out.
#
# The model's `differences` line is the only sentence in step 0 that says what
# the reference has and the product does not, and it is written before a penny
# is spent. On one run of the old pipeline it read "a slightly more defined
# V-neckline and visible seam piping along the edges" - and four of the ten
# candidates came back with a neckline seam and topstitching down the straps, at
# 15c each. The reference is supposed to contribute pose and nothing else, so
# ANY construction word in that line is worth naming: not a claim the run will
# fail, just the one early warning available.
CONSTRUCTION_TERMS = (
    "seam", "seams", "seaming", "piping", "stitch", "stitching", "topstitch",
    "topstitching", "topstitched", "panel", "panels", "panelling", "paneling",
    "neckline", "collar", "placket", "binding", "bound edge", "trim", "dart",
    "darts", "gusset", "pocket", "pockets", "zip", "zipper", "button",
    "buttons", "closure", "logo", "label", "mesh", "cutout", "cut-out",
    "overlay", "elastic", "hem", "cuff", "cuffs", "waistband", "drawstring",
    "hood", "rib", "ribbing", "yoke", "pleat", "pleats", "ruffle", "frill",
)


def construction_terms(text: str | None) -> list[str]:
    """Which construction words a differences line uses, in the order given."""
    if not text:
        return []
    low = str(text).lower()
    found: list[str] = []
    for t in CONSTRUCTION_TERMS:
        # Whole words only: 'seam' must not fire on 'seamless', which is a
        # fabric finish and the opposite of a construction difference.
        if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low) and t not in found:
            found.append(t)
    return found


# ---------------------------------------------------------------------------
# Installing the winner
# ---------------------------------------------------------------------------

def install(src: Path, dst: Path, greyscale: bool = True,
            silhouette: bool = False,
            match_to: Path | None = None) -> tuple[bool, str, dict]:
    """Write the winner to dst. Returns (changed, description, tone record).

    `match_to` is the source photo the reference garment's lightness is moved
    to - see common.prepare_reference_image(). None leaves the tone alone.

    Skips the write when the bytes would be identical, so a re-run does not
    churn the mtime - anything fingerprinting inputs reads an mtime that moved
    for no reason as a changed input.

    `silhouette` installs a line drawing of the reference's outline instead of
    the photograph. Same pose, same arrangement, no construction to copy at all
    - see common.outline_map().
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_outline = None
    if silhouette:
        tmp_outline = dst.with_name(f".{dst.stem}.outline.jpg")
        C.outline_map(src, tmp_outline)
        src = tmp_outline
    # Hidden, so a temp file left by a crash is not mistaken for a reference.
    tmp = dst.with_name(f".{dst.stem}.tmp.jpg")
    if silhouette:
        with Image.open(src) as im:
            im.convert("L").save(tmp, quality=95, subsampling=0)
        tone = {"applied": False, "why": "outline map - nothing to re-tone"}
    else:
        tone = C.prepare_reference_image(src, tmp, match_to, greyscale=greyscale)
    changed = not (dst.exists() and C.md5(dst) == C.md5(tmp))
    if changed:
        tmp.replace(dst)
    else:
        tmp.unlink()
    if tmp_outline and tmp_outline.exists():
        tmp_outline.unlink()
    with Image.open(dst) as check:
        return changed, f"{check.width}x{check.height} mode={check.mode}", tone


# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------

def calibrate(recs: list[dict], colour_weight: float) -> int:
    """Score every library image against every other and print the spread.

    The threshold cannot be inherited. The old pipeline's 90 and 95 were
    measured against a bra-shaped form with different fields and different
    weights, and a number that came from somewhere else is exactly the kind of
    constant this project has been bitten by. So: describe the library once,
    then read the threshold off the library itself.

    Pairs are split by whether they share a garment_type word, which is a rough
    stand-in for "should these two match". The useful output is the gap - the
    threshold belongs above where same-family pairs sit and below where
    different-family ones do. If those two distributions overlap heavily, no
    threshold will separate them and the form needs work, which is worth knowing
    before a run depends on it.

    Costs nothing after --index: every record is already on disk.

    Tone is not part of this. It never enters the gate - see TONE_WEIGHT_DEFAULT
    - so the threshold read off here is the one the gate actually uses.
    """
    same: list[float] = []
    diff: list[float] = []
    dq = 0
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if disqualify(a, b):
                dq += 1
                continue
            s, _ = score(a, b, colour_weight)
            (same if _words(a.get("garment_type", "")) &
             _words(b.get("garment_type", "")) else diff).append(s)

    total = len(recs) * (len(recs) - 1) // 2
    print(f"\ncalibration over {len(recs)} images, {total} pairs")
    print(f"  {dq} pair(s) disqualified outright and never scored")

    def spread(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"  {name:<28} none")
            return
        xs = sorted(xs)
        def pc(p):
            return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]
        print(f"  {name:<28} n={len(xs):<5} "
              f"min {xs[0]:5.1f}  p50 {pc(50):5.1f}  p90 {pc(90):5.1f}  "
              f"p95 {pc(95):5.1f}  p99 {pc(99):5.1f}  max {xs[-1]:5.1f}")

    spread("same garment type", same)
    spread("different garment type", diff)

    if same and diff:
        floor = sorted(same)[max(0, int(0.10 * len(same)))]
        ceiling = sorted(diff)[min(len(diff) - 1, int(0.95 * len(diff)))]
        print(f"\n  90% of same-type pairs score at or above {floor:.1f}")
        print(f"  95% of different-type pairs score at or below {ceiling:.1f}")
        if ceiling < floor:
            print(f"  -> the two separate. Put --threshold between "
                  f"{ceiling:.0f} and {floor:.0f}.")
        else:
            print(f"  -> they OVERLAP by {ceiling - floor:.1f} points. No "
                  f"threshold separates them cleanly; a high one (above "
                  f"{ceiling:.0f}) trades recall for precision and leans on "
                  f"stage C to catch the rest.")
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes:  0 installed   2 no match   1 broke")
    ap.add_argument("--query", type=Path, default=None,
                    help="the garment photo the reference has to match. "
                         "Defaults to <run>/archive/source_clean.jpg, the "
                         "SEGMENTED image - so the query and the library are "
                         "both a garment on a white plate.")
    ap.add_argument("--query-raw", action="store_true",
                    help="declare that --query has NOT been segmented. "
                         "Recorded in reference_selection.json; the score is "
                         "degraded because the room and the hang tag are being "
                         "measured as part of the garment.")
    ap.add_argument("--library", type=Path, default=C.REFERENCE_LIBRARY)
    ap.add_argument("--run", type=Path, default=None,
                    help="run folder for the record and the contact sheet "
                         "(default: this session's folder)")
    ap.add_argument("--install-to", type=Path, default=None,
                    help=f"where the winner is written "
                         f"(default <run>/archive/{CANON})")
    ap.add_argument("--threshold", "--min-score", type=float,
                    default=DEFAULT_THRESHOLD, metavar="SCORE",
                    help=f"score a library image must reach to be installed "
                         f"(default {DEFAULT_THRESHOLD:.0f}). PROVISIONAL - it "
                         f"is not measured against your library. Run "
                         f"--calibrate and set it from the numbers.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="how many survivors go to the model in stage C")
    ap.add_argument("--colour-weight", "--color-weight", dest="colour_weight",
                    type=float, default=COLOUR_WEIGHT_DEFAULT,
                    help=f"0 ignores colour entirely (default "
                         f"{COLOUR_WEIGHT_DEFAULT}: the winner is desaturated, "
                         f"so colour is a tiebreak, not a driver)")
    ap.add_argument("--tone-weight", type=float, default=TONE_WEIGHT_DEFAULT,
                    metavar="POINTS",
                    help=f"how many score points a candidate loses in the "
                         f"RANKING for being {TONE_SPAN_L:.0f} L* or more "
                         f"from the garment's lightness (default "
                         f"{TONE_WEIGHT_DEFAULT}). Never touches the "
                         f"threshold. 0 ignores tone in the order.")
    ap.add_argument("--no-tone-match", action="store_true",
                    help="install the reference at its own lightness. By "
                         "default the reference garment is scaled in linear "
                         "light to the query garment's mean L*, because "
                         "greyscale removes hue and not tone, and tone is "
                         "copied off image 2 just as readily.")
    ap.add_argument("--colour", "--color", dest="colour", action="store_true",
                    help="install the reference in colour; the default "
                         "desaturates it so it cannot act as a colour target")
    ap.add_argument("--silhouette", action="store_true",
                    help="install a line drawing of the winner's OUTLINE "
                         "instead of the photograph. It carries the pose and no "
                         "construction at all, so there is nothing for the "
                         "generator to copy across. LOOK AT THE RESULT: "
                         "common.outline_map builds the mask from tone, so a "
                         "boldly striped or colour-blocked reference has its "
                         "dark bands punched out as holes and the outline comes "
                         "back striped. Fine on a solid reference, wrong on that "
                         "one - pick a solid runner-up instead.")
    ap.add_argument("--no-model-veto", action="store_true",
                    help="accept the top-scoring candidate even when the model "
                         "answers 'none' in stage C")
    ap.add_argument("--compare-mode", choices=["multi", "composite", "none"],
                    default="multi")
    ap.add_argument("--index", action="store_true",
                    help="describe every library image into the cache and exit. "
                         "Run this after adding to the library, so a live run "
                         "never pays the describe cost.")
    ap.add_argument("--calibrate", action="store_true",
                    help="score every library image against every other and "
                         "print the spread, to choose --threshold from data")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and report, install nothing")
    ap.add_argument("--max-dim", type=int, default=1024,
                    help="longest edge sent to the model")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="parallel requests; keep low for a single-GPU server")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="")
    a = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    library = a.library.resolve()
    if not library.is_dir():
        print(f"no reference library at {library}", file=sys.stderr)
        return 1
    lib = library_images(library)
    if not lib:
        print(f"reference library {library} holds no images", file=sys.stderr)
        return 1

    client = Client(a.base_url, a.model, a.timeout)
    try:
        model = client.resolve_model()
    except Exception as e:  # noqa: BLE001
        print(f"cannot reach the vision server at {client.base_url}: {e}",
              file=sys.stderr)
        return 1

    # --- describe the library ---------------------------------------------
    if a.index or a.calibrate:
        print(f"describing {len(lib)} image(s) in {library.name}/  "
              f"[{model}]")
        t0 = time.time()
        recs, failures, cached = describe_all(client, lib, a.concurrency,
                                              a.max_dim, a.max_tokens)
        print(f"  {len(lib)} image(s) in {time.time() - t0:.1f}s "
              f"({cached} cached, {len(lib) - cached} sent)"
              + (f", {len(failures)} FAILED" if failures else ""))
        if a.calibrate:
            return calibrate(recs, a.colour_weight)
        return 1 if failures and not recs else 0

    run = (a.run or C.session_run_dir()).resolve()
    run.mkdir(parents=True, exist_ok=True)
    query = (a.query or (run / "archive" / "source_clean.jpg")).resolve()
    if not query.exists():
        print(f"query not found: {query}", file=sys.stderr)
        return 1

    print(f"reference selection: {query.name} vs {library.name}/ "
          f"({len(lib)} images)  [{model}]")
    if a.query_raw:
        # Every library image is a garment on a white plate. Scoring a raw phone
        # photo against them charges the query for its room and its hang tag,
        # and that score is the number the whole run is anchored to. Degraded,
        # not dead - but it has to be visible here and in the receipt, or a raw
        # run is indistinguishable from a clean one.
        print("  WARNING: matched against an unsegmented query - the "
              "background and any tag are being scored as part of the garment.")

    # --- stage A -----------------------------------------------------------
    print("\nstage A: describing")
    t0 = time.time()
    q_attrs, q_cached, q_secs = describe(client, query, a.max_dim, a.max_tokens)
    print(f"    query  {'cache' if q_cached else 'MODEL'} {q_secs:5.1f}s  "
          f"{query.name}")
    print(f"           -> {q_attrs.get('garment_type')} / "
          f"{q_attrs.get('body_region')} / sleeve {q_attrs.get('sleeve')} / "
          f"leg {q_attrs.get('leg')} / {q_attrs.get('front_opening')} front / "
          f"{q_attrs.get('color_name')}")
    recs, failures, cached = describe_all(client, lib, a.concurrency,
                                          a.max_dim, a.max_tokens)
    print(f"           {len(lib)} library image(s) in {time.time() - t0:.1f}s "
          f"({cached} cached, {len(lib) - cached} sent)"
          + (f", {len(failures)} failed" if failures else ""))
    if not recs:
        print("nothing in the library could be described", file=sys.stderr)
        return 1

    # --- stage B -----------------------------------------------------------
    # The query's tone is measured off the segmented photo. A raw photo's mask
    # takes in the bench and the wall, so under --query-raw the number is not
    # trusted: tone drops out of the ranking and nothing is re-toned.
    q_tone = ({"found": False, "L": None, "why": "query is unsegmented"}
              if a.query_raw else tone_of(query))
    ranked: list[dict] = []
    excluded: list[dict] = []
    for r in recs:
        why = disqualify(q_attrs, r)
        if why:
            excluded.append({"file": r.get("_file"), "why": why})
            continue
        s, parts = score(q_attrs, r, a.colour_weight)
        t = tone_of(Path(r["_path"]))
        dL = (round(t["L"] - q_tone["L"], 1)
              if t.get("found") and q_tone.get("found") else None)
        ranked.append({**r, "score": s, "parts": parts, "tone_L": t.get("L"),
                       "tone_dL": dL, "rank": rank_score(s, dL, a.tone_weight)})
    # Ordered by rank - score less the tone penalty - so the labels the model
    # sees, and the default pick, prefer the match that needs least re-toning.
    ranked.sort(key=lambda r: -r["rank"])
    for i, r in enumerate(ranked):
        r["label"] = chr(ord("A") + i) if i < a.top_k else ""

    print(f"\nstage B: scoring (colour weight {a.colour_weight}, tone weight "
          f"{a.tone_weight}, threshold {a.threshold:.0f})")
    print(f"  {len(excluded)} of {len(recs)} disqualified outright"
          + (f" - e.g. {excluded[0]['file']}: {excluded[0]['why']}"
             if excluded else ""))
    if q_tone.get("found"):
        print(f"  query garment L* {q_tone['L']:.1f} (grey {q_tone['grey']:.0f}); "
              f"dL* below is each candidate's garment against it")
    else:
        print(f"  query garment tone not measured ({q_tone.get('why')}) - "
              f"tone is out of the ranking")
    if not ranked:
        print("\nNO REFERENCE - every library image was disqualified. The "
              "library holds nothing of this kind of garment.")
    for i, r in enumerate(ranked[:max(a.top_k, 8)], 1):
        print(f"  {i:>2}. {r['score']:6.1f} "
              f"{'PASS' if r['score'] >= a.threshold else '  - '}  "
              f"dL* {fmt_dL(r['tone_dL'])} -> rank {r['rank']:5.1f}  "
              f"{str(r['_file'])[:40]:<40} {r.get('garment_type', '?')}, "
              f"{r.get('silhouette', '?')}, {r.get('fabric_finish', '?')}")

    # A candidate below the threshold is not a match, so it is not eligible to
    # win stage C either. Gate on SCORE before spending the comparison call;
    # the survivors keep their rank order.
    qualifying = [r for r in ranked if r["score"] >= a.threshold][:a.top_k]
    best = max(ranked, key=lambda r: r["score"]) if ranked else {}

    # --- stage C -----------------------------------------------------------
    verdict = None
    if qualifying and a.compare_mode != "none":
        # Run even for a single survivor: under a strict threshold the useful
        # question is "is this actually a match", not "which is closest".
        print(f"\nstage C: {'confirming the 1 candidate' if len(qualifying) == 1 else f'head-to-head on the {len(qualifying)} candidates'} "
              f"that cleared {a.threshold:.0f}")
        cands = [(r["label"], C.ensure_small(Path(r["_path"]), a.max_dim, SMALL))
                 for r in qualifying]
        q_small = C.ensure_small(query, a.max_dim, SMALL)
        tC = time.time()
        try:
            verdict = (compare_multi(client, q_small, cands)
                       if a.compare_mode == "multi"
                       else compare_composite(client, q_small, cands))
        except Exception as e:  # noqa: BLE001
            print(f"  {a.compare_mode} mode failed ({e}); retrying as composite")
            try:
                verdict = compare_composite(client, q_small, cands)
            except Exception as e2:  # noqa: BLE001
                print(f"  composite also failed: {e2}")
        if verdict:
            raw = str(verdict.get("pick", "")).strip().upper()
            by_label = {r["label"]: r for r in qualifying}
            print(f"  replied in {time.time() - tC:.1f}s")
            if raw in ("NONE", "", "NULL"):
                print(f"  pick:        none - rejected all {len(qualifying)}")
            else:
                hit = by_label.get(raw)
                print(f"  pick:        {raw}"
                      + (f"  -> {hit['_file']}" if hit else "  (not a label)"))
            print(f"  confidence:  {verdict.get('confidence')}")
            print(f"  reason:      {verdict.get('reason')}")
            print(f"  differences: {verdict.get('differences')}")

    # --- decision ----------------------------------------------------------
    # Two independent gates, both must pass: the number clears the threshold,
    # and the model does not reject the survivors on sight.
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
        if vetoed and not a.no_model_veto:
            match_found, chosen = False, None
        elif vetoed:
            print("\n  model said none, but --no-model-veto is set; keeping "
                  "the top-scoring candidate")

    sheet = None
    if ranked:
        tiles = [("QUERY", C.ensure_small(query, a.max_dim, SMALL))]
        colours = {}
        for r in ranked[:a.top_k]:
            passed = r["score"] >= a.threshold
            lab = f"{r['label'] or '-'}  {r['score']:.0f}" + ("" if passed else "  x")
            colours[lab] = (30, 80, 200) if passed else (140, 140, 140)
            tiles.append((lab, C.ensure_small(Path(r["_path"]), a.max_dim, SMALL)))
        sheet = C.contact_sheet(tiles, run / "reference_match.jpg",
                                colours=colours)

    # Compared on _path, not _file: library_images() rglobs, so a library
    # somebody nested into folders can hold two files with the same basename and
    # the wrong one would be dropped from the ranking as "the winner".
    # None when nothing was chosen - there is no runner-up to a race with no
    # winner, and `closest` already carries the near miss.
    runner = (next((r for r in ranked if r["_path"] != chosen["_path"]), {})
              if chosen else {})
    bleed = {"flagged": False, "terms": [], "action": "none",
             "line": (verdict or {}).get("differences")}
    if chosen:
        terms = construction_terms((verdict or {}).get("differences"))
        if terms:
            bleed.update({"flagged": True, "terms": terms,
                          "action": ("silhouette installed, so nothing can be "
                                     "copied" if a.silhouette else "warned")})

    record = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "match_found": match_found,
        "query": str(query),
        "query_md5": C.md5(query),
        "query_cleaned": not a.query_raw,
        "query_attrs": q_attrs,
        "library_root": str(library),
        "library_count": len(lib),
        "described": len(recs),
        "disqualified": excluded,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "threshold": a.threshold,
        "colour_weight": a.colour_weight,
        "tone_weight": a.tone_weight,
        "query_tone": q_tone,
        "n_qualifying": len(qualifying),
        "model_vetoed": vetoed,
        "model_confidence": (verdict or {}).get("confidence"),
        "source": str(chosen["_path"]) if chosen else None,
        "score": round(chosen["score"], 1) if chosen else None,
        "closest": {"file": best.get("_file"),
                    "score": round(best["score"], 1) if best else None,
                    "dL": best.get("tone_dL")},
        "runner_up": {"file": runner.get("_file"),
                      "score": round(runner["score"], 1) if runner else None,
                      "dL": runner.get("tone_dL")},
        # The pick's tone against the query's, and what was done about it.
        # `tone_match` is filled in once the file is written.
        "tone": ({"reference_L": chosen.get("tone_L"),
                  "query_L": q_tone.get("L"), "dL": chosen.get("tone_dL"),
                  "rank": round(chosen["rank"], 1)} if chosen else None),
        "tone_match": None,
        "reason": (verdict or {}).get("reason"),
        "differences": (verdict or {}).get("differences"),
        "construction_risk": bleed,
        "silhouette": bool(a.silhouette),
        "greyscale": not a.colour,
        "contact_sheet": str(sheet) if sheet else None,
        "describe_failures": failures,
        "installed": None,
        "installed_md5": None,
    }

    if not match_found:
        print("\nNO REFERENCE SELECTED - nothing in the library matched this "
              "garment.")
        if vetoed:
            print(f"  {len(qualifying)} candidate(s) cleared "
                  f"{a.threshold:.0f} but the model rejected them on sight")
            print("  -> --no-model-veto trusts the score alone")
        elif best:
            print(f"  closest    {best.get('_file')} at {best['score']:.1f} "
                  f"(needed {a.threshold:.0f})")
        if sheet:
            print(f"  near misses {sheet}")
        print("  nothing was installed.")
        (run / "reference_selection.json").write_text(
            json.dumps(record, indent=2, default=str))
        why = ("the model vetoed every candidate" if vetoed
               else f"best {best.get('score', 0):.1f} < {a.threshold:.0f}")
        C.log(run, f"reference NOT selected ({why})")
        return 2

    src = Path(chosen["_path"])
    dst = (a.install_to or (run / "archive" / CANON)).resolve()

    if bleed["flagged"]:
        print(f"\nCONSTRUCTION BLEED RISK - the pick differs from the product "
              f"in {len(bleed['terms'])} construction term(s): "
              f"{', '.join(bleed['terms'])}")
        print(f"  differences: {bleed['line']}")
        print("  The reference is a LAY reference. Anything it shows that the "
              "product does not have can be copied into the generation.")
        if not a.silhouette:
            print("  These words go into the agent's opening brief. Stronger, "
                  "if it keeps happening: re-run with --silhouette, which "
                  "installs an outline and leaves nothing to copy.")

    if a.dry_run:
        print(f"\nDRY RUN - would install {src} -> {dst}"
              + ("" if a.colour else " (greyscale)")
              + (" (as an outline map)" if a.silhouette else "")
              + ("" if (a.colour or a.silhouette or a.no_tone_match
                        or not q_tone.get("found"))
                 else f" (garment re-toned to L* {q_tone['L']:.1f})"))
        (run / "reference_selection.json").write_text(
            json.dumps(record, indent=2, default=str))
        return 0

    changed, desc, tone = install(
        src, dst, greyscale=not a.colour, silhouette=a.silhouette,
        match_to=None if (a.no_tone_match or a.query_raw) else query)
    if a.no_tone_match:
        tone["why"] = "--no-tone-match"
    elif a.query_raw and not tone.get("applied"):
        tone["why"] = "query is unsegmented, its garment tone cannot be trusted"
    record["tone_match"] = tone

    # The library asset, byte for byte, beside the desaturated one. Copied and
    # never converted: the library is read-only as far as this tool is
    # concerned, and this is the only copy of the reference's real colour the
    # run folder will hold. Without it, "what colour was the reference" is a
    # question that can only be answered by going back to the library and
    # working out which file was used.
    original = dst.with_name(C.REFERENCE_ORIGINAL)
    try:
        shutil.copy2(src, original)
        record["original"] = str(original)
    except OSError as e:
        print(f"  (could not keep a colour copy of the reference: {e})")
        record["original"] = None

    record["installed"] = str(dst)
    record["installed_md5"] = C.md5(dst)
    record["installed_desc"] = desc
    record["rewritten"] = changed
    record["source_md5"] = C.md5(src)
    (run / "reference_selection.json").write_text(
        json.dumps(record, indent=2, default=str))

    print("\nREFERENCE SELECTED")
    print(f"  source       {src.name}")
    print(f"  score        {chosen['score']:.1f}/100"
          + (f"   model confidence {verdict['confidence']}"
             if verdict and verdict.get("confidence") is not None else ""))
    print(f"  installed    {dst}"
          + ("   <- OUTLINE MAP, not the photograph" if a.silhouette else ""))
    print(f"               {desc}  md5:{record['installed_md5'][:8]}"
          + ("" if changed else "  (already identical, not rewritten)"))
    if record.get("original"):
        print(f"  colour copy  {Path(record['original']).name}  "
              f"(the library asset, unmodified)")
    if tone.get("applied"):
        print(f"  tone         garment grey {tone['grey_before']:.0f} -> "
              f"{tone['grey_after']:.0f}, to match the source's "
              f"{tone['target_grey']:.0f}  (x{tone['scale']:.3f} in linear "
              f"light)"
              + (f"   SHORT by {tone['shortfall']:.0f} levels"
                 if tone.get("shortfall") else ""))
    else:
        print(f"  tone         left alone - {tone.get('why')}")
    if runner:
        print(f"  runner-up    {runner.get('_file')} ({runner['score']:.1f})")
    if record["differences"]:
        print(f"  differences  {record['differences']}")
    if bleed["flagged"]:
        print(f"  bleed risk   {', '.join(bleed['terms'])}")
    if sheet:
        print(f"  contact      {sheet}")
    print(f"  record       {run / 'reference_selection.json'}")

    C.log(run, f"reference {src.name[:26]} ({chosen['score']:.0f}/100)"
               + ("" if not a.query_raw else " vs RAW query")
               + (f", toned {tone['grey_before']:.0f}->{tone['grey_after']:.0f}"
                  if tone.get("applied") else "")
               + (f", bleed: {','.join(bleed['terms'][:3])}"
                  if bleed["flagged"] else "")
               + (" [outline]" if a.silhouette else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
