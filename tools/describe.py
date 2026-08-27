#!/usr/bin/env python3
"""Inventory the garment's construction, so the re-lay model cannot invent any.

    python tools/describe.py --run runs/<stamp>

Runs the self-hosted Qwen vision model over the CLEANED source and writes
`archive/garment_description.md`. `generate.py` then appends it to every prompt
automatically, so each draw is anchored to the same written inventory rather
than to whatever the re-lay model infers from the pixels that pass.

This step used to call Gemini through fal's `openrouter/router/vision` while
every other vision step in the project - step 0's attribute read, grading, the
harness's own view_image - already talked to the local server. That split was
history, not a decision: this file was written against the hosted route and
never revisited. It now uses the same server as the rest, so the pipeline has
one vision model, no per-run charge and no outside dependency.

The prompts below still carry the failure modes of the hosted models in their
comments. Those are kept deliberately - they record what a VLM asked about
garment construction gets wrong, which is a property of the task rather than of
any one model, and they are why the audit at the bottom of this file exists.

Why this exists: invented and lost construction has been the most persistent
failure on this project - a candidate growing topstitching the product does not
have, or smoothing away a seam it does. A model told "there are exactly two
pockets, no zip, no drawcord" has far less room to hallucinate than one told
"keep the construction".

The negative inventory is the important half. Listing what is ABSENT suppresses
invention more reliably than listing what is present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import common as C
from vision import (Client, DEFAULT_BASE_URL, DEFAULT_MODEL, ensure_small,
                    image_part, text_part)

# The cleaned source is a 4K plate. The local server takes the image inline as
# base64 rather than by URL, so it is downscaled first - but to 1536, not the
# 1024 the other vision steps use. Those steps judge silhouette, colour and
# lay, which survive a small frame; this one is asked to resolve topstitching
# and seam finishes, and a missed stitch line here becomes a NOT-PRESENT claim
# telling the generator to delete real construction.
MAX_DIM = 1536

# Transcription beats inference. Five attempts at getting a positive seam list
# out of a VLM looking at a photograph all fabricated - gemini invented an
# inseam and a side seam, claude invented seven panels and a seam down each leg
# - because a model asked what a garment has answers from what that CATEGORY
# usually has. A spec sheet removes the guess: this one states "No inseam with
# oval gusset", which is exactly the seam both models drew.
BOM_ASK = """\
This is a garment construction spec sheet. Transcribe EVERY callout verbatim,
one per line, exactly as written - stitch types, needle counts, seam allowances
and all.

Then output two sections:

**CONSTRUCTION** - the callouts rewritten as a plain list of what the garment
is built with, keeping the stitch specification for each.

**NOT PRESENT** - a comma-separated list of construction this sheet says the
garment does NOT have. Read the callouts carefully: a note reading "no inseam"
means the garment has no inseam. Add any standard feature the sheet would have
called out and did not.

Transcribe and read only. Infer nothing that is not written on the sheet.
"""

SYSTEM = ("You are a technical garment analyst writing a construction spec for "
          "a photo retoucher. You describe ONLY what is visible in this single "
          "photograph. You are looking at one side of the garment and cannot "
          "see the other - never mention a back seam, back panel, back yoke or "
          "back pocket unless it is genuinely visible in this frame. You never "
          "infer construction from what garments of this type usually have. You "
          "never describe the background, the framing or the lay.")

# Appended when the source has not been pre-cleaned. Since the object-removal
# endpoint went 403 every run is in this state, so the model is looking at the
# tag, the pins and whatever the garment is hanging from. Left unsaid it writes
# them up as construction, and a prompt built from that asks an image model to
# keep them - which is how a run once shipped four candidates still wearing the
# hang tag.
DIRTY = (" This photograph has NOT been retouched. It may still show a hang "
         "tag, price ticket, safety or straight pins, clips, a hanger, tissue "
         "or stuffing. None of those are part of the garment. Never count them "
         "as construction, as an applied element, or as a panel or seam - they "
         "belong only in the TO REMOVE list.")

# The candidate vocabulary is DERIVED per garment, not hardcoded. A fixed list
# is legwear wearing a disguise: "side seam down the leg" is meaningless on a
# bra, and at scale every category would need its own hand-written list. So the
# model is asked what features garments of THIS type commonly have, and then
# adjudicates its own list against the actual photograph.
VOCAB_ASK = """\
This is a flat product photograph of a single garment.

Name the garment type in three words. Then list 30 to 40 CONSTRUCTION FEATURES
that garments of this type commonly have - closures, seams by location, panels,
pockets, trims, hardware, applied branding, edge finishes. Include the ordinary
and the optional. Do not look at whether this particular garment has them; you
are building a checklist for the category.

Output only a comma-separated list on one line, nothing else.
"""

# A minimal fallback if the vocabulary call fails. Deliberately garment-neutral.
FALLBACK = ["zip or zipper", "buttons", "snaps", "drawcord or drawstring",
            "elastic cord or toggle", "mesh or ventilation panels",
            "sheer panels", "contrast piping or binding",
            "contrast topstitching in another colour", "reflective trim",
            "colour blocking", "printed graphic", "embroidery",
            "external brand logo or wordmark", "visible label or tab outside",
            "seam taping", "gusset", "vents or slits", "ruching or gathering",
            "visible lining", "buckles or hardware", "grommets or eyelets",
            "cut-out detail", "pockets", "cuffs or ribbed trims"]

ASK = """\
This is a flat product photograph of a single garment on a white background.

## Part 1 - what it has

Terse markdown, only what you can actually see:

**Garment** - what it is, in three words.
**Colour and fabric** - colourway, and visible character (smooth knit, ribbed,
brushed, matte, sheen).
**Seams** - only seams you can actually SEE as a line in this image. Say where
each runs and how it is finished. **Do not list a seam because garments of this
type normally have one** - a run listed an inseam and a side seam on leggings
whose legs are visibly unbroken. If you are not looking straight at it, leave it
out.
**Panels** - how many are visible, and where the divisions fall. Count only
panels you can see in this frame.
**Openings and pockets** - how many, exactly where, how each is finished.
**Bands, waistband, cuffs, hems** - construction and depth, and whether
topstitched.
**Applied elements** - logos, labels, prints, hardware, and where. If none, say
none.

## Part 2 - TO REMOVE

Items ATTACHED TO or SITTING ON the garment that are not part of it, and must
not appear in the final image: hang tags, swing tickets, price tickets, barcode
or size stickers, safety pins, straight pins, clips, clamps, hangers, hanger
hooks, tissue, stuffing, foam inserts, props, hands, mannequin parts.

The test is whether it was ADDED FOR THE PHOTO. If it would still be on the
garment in a customer's wardrobe, it is part of the garment and does NOT go on
this list. In particular, a sewn-in woven brand label, neck label, size label or
care label IS part of the garment - keep it, and do not list it here.

Under the heading `**TO REMOVE**`, list each one you can see, with where it is,
one per line. Write `none` if the garment is on its own - that is the normal
answer and you should not hunt for something to list.

These are NOT construction. Do not list any of them in Part 1 or Part 3, and do
not describe them as though they were part of the garment.

## Part 3 - NOT PRESENT

Go through this list one item at a time and decide for each whether it is
visible on THIS garment. Then output, under the heading `**NOT PRESENT**`, a
single comma-separated list naming every item that is absent. Do not skip items,
do not summarise, and do not add anything to that list that you can actually
see.

If an item is genuinely ambiguous, leave it out of the NOT PRESENT list and note
it under `**UNCERTAIN**` instead.

Items to adjudicate:
{checklist}

You are seeing ONE side of this garment. Do not describe or infer the side
facing away from the camera. If you cannot tell whether a seam continues around
the other side, say only what you can see.

Do not describe the background, framing, pose, wrinkles, or how the garment is
laid out. Construction only.
"""

# Terms unambiguous enough that appearing in BOTH halves is a real contradiction.
# Pocket TYPES are excluded on purpose - a side-seam pocket is legitimately
# "not a patch pocket, not a welt pocket", so those co-occur harmlessly.
#
# Kept for the headline warning only. The real check is audit_absent() below,
# which is derived from the document rather than from this list - a hand-written
# vocabulary only catches the contradictions somebody thought of, and the one
# that cost this project was "Pearl embellishments", which is not on it.
CONTRADICTION_TERMS = [
    "zip", "button", "snap", "drawcord", "drawstring", "belt loop", "mesh",
    "sheer", "stripe", "piping", "reflective", "embroidery", "logo", "hood",
    "collar", "thumbhole", "gusset", "lining", "grommet", "buckle",
]

# Attributes step 0 measured that name construction, and are specific enough to
# contradict a NOT-PRESENT claim. Deliberately not garment_type - "bra" is three
# letters and substring-matches half the language - and deliberately not the
# boolean fields, which are only used when TRUE: `adjusters: false` makes
# "Adjustable straps" a CORRECT absence, and dropping it would be the mistake
# this function exists to prevent, in reverse.
ATTR_FIELDS = ("strap_style", "closure", "neckline", "padding", "band",
               "support_level", "coverage")

# Words too common to carry meaning inside a garment feature name.
STOP = {"the", "and", "with", "for", "this", "that", "from", "into", "onto",
        "detail", "details", "style", "styles", "type", "types", "fabric",
        "front", "back", "side", "outer", "inner", "left", "right", "medium",
        "standard", "smooth", "matte", "wide", "unknown", "none", "true",
        "false", "visible", "small", "large"}

# Properties of a textile that no photograph can show. A model asked to
# adjudicate them answers from the category, not the image, so "does NOT have
# moisture-wicking fabric" is a guess dressed as an observation - and it is
# being sent to an image generator, which cannot draw the absence of wicking
# either way. Noise, not signal, so it does not travel.
UNSEEABLE = ("wicking", "moisture", "breathable", "compression", "stretch",
             "quick dry", "quick-dry", "antimicrobial", "anti-odor",
             "anti-odour", "uv protection", "spf", "recycled", "sustainable",
             "seamless construction")


# Phrases that assert which way round the garment is, rather than what it is
# built from. The describe pass is looking at one photograph and guessing:
# runs/20260820_112558 came back "two visible seams on the upper back" and "two
# back panels forming the criss-cross straps" for a garment photographed front
# up. Read on its own that is a harmless mistake; copied into a prompt it is an
# instruction, and seven of that run's ten candidates came back showing the
# reverse face.
#
# Location words are NOT here on purpose. "Back panel" describes a panel that
# exists whichever side is up, and stripping it would throw away real
# construction - the half of this file that suppresses invention. Only a claim
# about the VIEWPOINT travels badly, so only a claim about the viewpoint goes.
ORIENTATION = ("shown from", "viewed from", "seen from", "from the back",
               "from the front", "from behind", "back view", "front view",
               "rear view", "reverse side", "inside out", "wrong side",
               "we are looking at", "this is the back", "this is the front",
               "the interior is visible", "facing away")


def strip_orientation(text: str) -> tuple[str, list[str]]:
    """Remove sentences that claim which face of the garment is toward the
    camera. Returns (text, removed)."""
    kept, removed = [], []
    for line in text.splitlines():
        # Sentence-wise, so one bad clause does not cost a whole bullet of
        # genuine construction. Splitting on '. ' keeps the markdown intact.
        parts = re.split(r"(?<=\.)\s+", line)
        good = [p for p in parts if not any(o in p.lower() for o in ORIENTATION)]
        removed += [p.strip() for p in parts if p not in good and p.strip()]
        kept.append(" ".join(good) if good else "")
    return "\n".join(kept), removed


def _words(s: str) -> list[str]:
    """Lowercase alphabetic tokens, crudely singularised."""
    out = []
    for w in re.findall(r"[a-z]+", str(s).lower()):
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def _squash(s: str) -> str:
    return "".join(re.findall(r"[a-z]+", str(s).lower()))


def _phrase_in(item: str, text: str) -> bool:
    """Is `item` present in `text` as a contiguous phrase, ignoring plurals?

    Contiguous, not word-by-word: a document that says "shoulder straps" and
    "armhole seams" contains both words of "shoulder seams" and claims neither.
    """
    a, b = _words(item), _words(text)
    a = [w for w in a if w not in STOP]
    if not a:
        return False
    return any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


def removals(text: str) -> list[str]:
    """The TO REMOVE list: what is in the photo but is not the garment.

    This exists because the pre-clean does not run any more - fal's
    object-removal endpoint is restricted - so the tag, pins and hanger are
    still in the pixels the generator is handed. Naming them explicitly is the
    replacement: the prompt asks for them to be taken out, rather than them
    having been erased beforehand.

    Read only. Anything here is a non-garment object, never construction.
    """
    up = text.upper()
    if "TO REMOVE" not in up:
        return []
    tail = text[up.index("TO REMOVE"):].split("\n", 1)[-1]
    # Stops at whatever section follows, at this file's own audit comment, and
    # at a markdown rule, so re-reading an annotated file cannot parse the
    # annotation back in.
    for stop in ("## Part", "**NOT PRESENT", "**UNCERTAIN", "<!--", "\n---"):
        tail = tail.split(stop)[0]
    out = []
    for line in tail.splitlines():
        item = line.strip(" .-*\t")
        # A sewn-in label is part of the garment and stays. The prompt says so,
        # and the model still lists it about half the time - runs/20260827_101946
        # named "White woven size label (visible inside the back neckline)" and
        # the prompt duly told the generator to erase it, which is the same
        # class of damage as a false NOT-PRESENT claim. Wording did not hold at
        # temperature 0.2, so it is enforced here instead.
        low = item.lower()
        if any(k in low for k in ("woven label", "woven size", "neck label",
                                  "size label", "care label", "brand label",
                                  "sewn-in", "sewn in", "interior label",
                                  "inside label", "wash label")):
            continue
        # The model writes the section twice - once as the `## Part 2 - TO
        # REMOVE` heading and once as the `**TO REMOVE**` marker - and the
        # index above lands on the first. Without this the marker line itself
        # is parsed as an item, and the prompt asks the generator to remove a
        # thing called "TO REMOVE".
        if _squash(item) in ("toremove", "") or item.startswith("#"):
            continue
        if 2 < len(item) < 120 and item.lower() not in ("none", "none.", "n/a"):
            out.append(item)
    return out


def _strip_removals(head: str) -> str:
    """Drop the TO REMOVE block out of the positive half before it is used to
    contradict a NOT-PRESENT claim. Both sections sit above NOT PRESENT, and
    without this a tag named in both halves reads as "Part 1 says it IS there"
    - true of the photograph, but the garment does not have a hang tag, and the
    reason recorded would be wrong."""
    up = head.upper()
    if "TO REMOVE" not in up:
        return head
    i = up.index("TO REMOVE")
    rest = head[i:]
    j = rest.find("## Part")
    return head[:i] + (rest[j:] if j != -1 else "")


def audit_absent(text: str, attrs: dict | None = None) -> dict:
    """Which NOT-PRESENT claims are safe to send to the generator.

    The negative inventory is the half that suppresses invention, and it is
    also the half that can do the most damage: every item on it becomes "the
    garment specifically does NOT have this" in a prompt, and a false entry
    tells an image model to delete real construction.

    On runs/20260819_205617 the list named `Pearl embellishments` while Part 1
    of the same document described two pearl embellishments on the straps and
    two more on the band; it named `Side seams` while Part 1 located the pearls
    "at the bottom of the side seams"; and it named `Racerback straps` and
    `Pullover style` for a garment step 0 had measured as strap_style
    'racerback', closure 'pullover'. All four were sent.

    Three ways an item is dropped, each recorded with its reason:
      * the positive half of the same document says it IS there
      * step 0 measured an attribute that says it is there
      * no photograph could show it either way (fabric performance claims)

    Nothing is added and nothing is rewritten - the model's own text stays as
    written. This only decides what travels.
    """
    up = text.upper()
    if "NOT PRESENT" not in up:
        return {"present": text, "absent": [], "keep": [], "dropped": []}
    # rindex, not index: the model writes the section twice, as a `## Part 3 -
    # NOT PRESENT` heading and again as the `**NOT PRESENT**` marker above the
    # list. Taking the first put the marker line inside the list, so the first
    # absence parsed came out as "NOT PRESENT**\nzip".
    head = _strip_removals(text[:up.index("NOT PRESENT")])
    tail = text[up.rindex("NOT PRESENT"):].split("\n", 1)[-1]
    # Stop at the UNCERTAIN section and at this function's own audit comment -
    # re-reading a file it has already annotated must not parse the annotation
    # back in as more claims.
    tail = tail.split("**UNCERTAIN")[0].split("<!--")[0].strip().lstrip("*- ")
    items = [i.strip(" .-\n*") for i in tail.split(",")]
    items = [i for i in items if 2 < len(i) < 80]

    tokens = {}
    for f in ATTR_FIELDS:
        v = (attrs or {}).get(f)
        if not isinstance(v, str):
            continue
        for w in re.findall(r"[a-z]+", v.lower()):
            if len(w) >= 5 and w not in STOP:
                tokens[w] = f

    keep, dropped = [], []
    for it in items:
        sq = _squash(it)
        why = None
        if _phrase_in(it, head):
            why = "the description's own Part 1 says it IS there"
        elif any(t in sq for t in tokens):
            t = next(t for t in tokens if t in sq)
            why = f"step 0 measured {tokens[t]} = '{(attrs or {}).get(tokens[t])}'"
        elif any(u in it.lower() for u in UNSEEABLE):
            why = "not visible in a photograph, so it was never adjudicated"
        (dropped.append((it, why)) if why else keep.append(it))
    return {"present": head, "absent": items, "keep": keep, "dropped": dropped}


def _vocab_once(client: Client, image: Path) -> list[str]:
    out = client.chat([image_part(ensure_small(image, MAX_DIM)),
                       text_part(VOCAB_ASK)],
                      max_tokens=600, temperature=0.3).strip()
    items = [x.strip(" .-") for x in out.replace("\n", ",").split(",")]
    return [x for x in items if 3 < len(x) < 70]


def vocabulary(client: Client, image: Path) -> list[str]:
    """Ask what features THIS category of garment commonly has.

    The prompt asks for 30 to 40 and the local model does not reliably give
    that. Two failures seen on real runs, in both directions: one answer parsed
    down to 3 usable items, and another ran to 136 by degenerating into
    variations on one word ("cup shaped, cup contoured, cup sculpted...").
    Neither is a hard error - both return text, so `items or FALLBACK` passes
    them straight through - and both damage the step, because the checklist
    IS the negative inventory. Three items means almost nothing gets
    adjudicated; 136 pads the absent list with filler for the generator to act
    on.

    So the length is checked rather than assumed: retry once, then top up from
    the neutral fallback, and cap. A short list is the worse of the two - it
    silently removes the half of this file that suppresses invention.
    """
    items = _vocab_once(client, image)
    if len(items) < 12:
        items = _vocab_once(client, image) or items
    if len(items) < 12:
        # Top up rather than replace: what the model did name is specific to
        # this garment and worth keeping ahead of the generic list.
        seen = {i.lower() for i in items}
        items += [f for f in FALLBACK if f.lower() not in seen]
    return items[:40]


def describe(client: Client, image: Path, vocab: list[str],
             dirty: bool = False) -> str:
    # Roomier than the hosted call's 1500: Part 1 plus an item-by-item verdict
    # on 30-40 checklist entries is a long answer, and truncation lands in the
    # NOT-PRESENT list - the half that matters. Local tokens are free, so there
    # is no reason to run this near the edge.
    return client.chat(
        [image_part(ensure_small(image, MAX_DIM)),
         text_part(ASK.format(checklist="\n".join(f"- {i}" for i in vocab)))],
        max_tokens=2000, temperature=0.2,
        system=SYSTEM + (DIRTY if dirty else "")).strip()


def from_bom(client: Client, bom: Path) -> str:
    """Read the construction off the spec sheet rather than guessing it."""
    # Larger than MAX_DIM: a spec sheet is dense small type and this is a
    # transcription task, so 1536 is how a "1/4in SS" callout becomes an
    # unreadable smudge. It still goes through ensure_small rather than in raw -
    # the sheet is typically a PNG, image_part() labels its bytes as JPEG, and
    # sips does the conversion that makes that label true.
    return client.chat([image_part(ensure_small(bom, 2048)),
                        text_part(BOM_ASK)],
                       max_tokens=2000, temperature=0.0).strip()


def query_attrs(run: Path) -> dict:
    """What step 0 measured about this garment, if it ran."""
    f = Path(run) / "reference_selection.json"
    try:
        return json.loads(f.read_text()).get("query_attrs") or {}
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return {}


def report_absent(out: Path, text: str, attrs: dict,
                  orientation: list[str] | None = None) -> dict:
    """Print the audit and record it in the file.

    NOT-PRESENT claims that are contradicted stay visible in the document - it
    is the record of what the model actually said - but they are listed in a
    comment as not sent, so reading the file and reading the prompt give the
    same answer. generate.py runs the same audit and sends only what survives.

    Orientation claims are different and are REMOVED from the body before it is
    written, not just withheld: the consumer of this file is a person or an
    agent writing a prompt from it, and "shown from the back" does its damage by
    being read, not by being appended. They are listed here so the removal is
    auditable.
    """
    a = audit_absent(text, attrs)
    lines = []
    if orientation:
        print(f"  REMOVED {len(orientation)} orientation claim(s) - which face "
              f"is toward the camera is not construction, and copied into a "
              f"prompt it flips the garment:")
        for s in orientation:
            print(f"    {s[:100]}")
        lines += [f"     orientation removed: {s}" for s in orientation]
    if a["absent"]:
        print(f"              NOT-PRESENT: {len(a['keep'])} of {len(a['absent'])} "
              f"claims will be sent")
        if a["dropped"]:
            print("  DROPPED as unsafe to send - a false absence tells the model "
                  "to delete real construction:")
            for item, why in a["dropped"]:
                print(f"    {item:<32} {why}")
            lines += [f"     {i} - {w}" for i, w in a["dropped"]]
    if lines:
        note = ("\n\n<!-- audit: removed from the body, or withheld from the "
                "generator\n" + "\n".join(lines) + "\n-->\n")
        out.write_text(out.read_text().rstrip() + note)
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--image", type=Path,
                    help="default <run>/archive/offset_upload.jpg, i.e. the "
                         "CLEANED source - describing the dirty original would "
                         "inventory the hang tag as construction")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="default: whichever model the server lists first")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--bom", type=Path,
                    help="construction spec sheet. Defaults to "
                         "inputs/Design_BOM.png when it exists. When present it "
                         "is the AUTHORITY and the photo is not asked to supply "
                         "construction at all.")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dirty-source", action="store_true",
                    help="the image still has its tag, pins and hanger. Tells "
                         "the model they are not construction and belong in "
                         "TO REMOVE. Set by prepare.py whenever the pre-clean "
                         "did not run, which is currently always.")
    a = ap.parse_args()

    run = a.run or C.session_run_dir()
    img = a.image or run / "archive" / "offset_upload.jpg"
    if not img.exists():
        return print(f"Not found: {img}. Run prepare.py first.") or 1
    out = a.out or run / "archive" / "garment_description.md"

    client = Client(a.base_url, a.model, a.timeout)
    try:
        model = client.resolve_model()
    except Exception as e:  # noqa: BLE001 - any failure to reach it reads alike
        # prepare.py runs this as a subprocess and only checks whether the file
        # appeared, so the reason has to be on stdout or it is lost.
        print(f"cannot reach the vision server at {client.base_url}: {e}\n"
              f"  start it, or set QWEN_BASE_URL / --base-url. Without it "
              f"there is no construction inventory and every prompt goes out "
              f"unanchored.")
        return 1

    bom = a.bom if a.bom is not None else C.INPUTS / "Design_BOM.png"
    if bom.exists():
        print(f"reading spec  {bom.name} via {model}  "
              f"<- AUTHORITATIVE, not inferred from the photo")
        text = from_bom(client, bom)
        if text:
            text, orient = strip_orientation(text)
            out.write_text(f"<!-- source: {bom.name} (spec sheet, transcribed) "
                           f"-->\n\n" + text + "\n")
            words = len(text.split())
            up = text.upper()
            print(f"              -> {out.name}  {words} words")
            if "NOT PRESENT" not in up:
                print("  WARNING: the sheet yielded no NOT-PRESENT section.")
            # A transcribed sheet is authoritative, but a transcription can
            # still contradict itself or the garment step 0 measured, and the
            # safe direction is the same either way: do not tell the model to
            # remove something that may be there.
            report_absent(out, text, query_attrs(run), orient)
            # Zero cents, and still logged: the run tally is also the record of
            # which steps ran, and a step that stops charging must not stop
            # appearing.
            C.log(run, f"construction from spec sheet, {words} words")
            return 0
        print("  spec sheet yielded nothing; falling back to the photograph.")

    print(f"describing    {img.name} via {model}  "
          f"<- INFERRED from the photo, no spec sheet found")
    vocab = vocabulary(client, img)
    print(f"              category checklist: {len(vocab)} features derived "
          f"for this garment type")
    text = describe(client, img, vocab, dirty=a.dirty_source)
    if not text:
        return print("The vision model returned nothing.") or 1

    text, orient = strip_orientation(text)
    out.write_text(text + "\n")
    words = len(text.split())
    print(f"              -> {out.name}  {words} words")
    rm = removals(text)
    if rm:
        print(f"              TO REMOVE: {len(rm)} non-garment item(s) in the "
              f"photo - the prompt will ask for these to be taken out:")
        for r in rm:
            print(f"    {r[:100]}")
    elif a.dirty_source:
        # Worth saying. The source is known to be unretouched, so "nothing to
        # remove" is a claim about the photograph, not a step that was skipped.
        print("              TO REMOVE: none found - the model says the "
              "garment is on its own in this frame.")
    up = text.upper()
    if "NOT PRESENT" not in up:
        print("  WARNING: no NOT-PRESENT section. That is the half that "
              "suppresses invention; re-run before generating.")
    else:
        tail = text[up.index("NOT PRESENT"):]
        named = sum(1 for i in vocab
                    if i.split(" or ")[0].split(",")[0].lower() in tail.lower())
        print(f"              NOT-PRESENT list covers {named}/{len(vocab)} "
              f"derived features")
        if named < len(vocab) * 0.4:
            print("  WARNING: the model adjudicated fewer than half the "
                  "checklist. The absent list is the anchor against invention.")

        # A feature named in both halves would tell the re-lay model to remove
        # something the garment actually has - worse than inventing one.
        head = text[:up.index("NOT PRESENT")].lower()
        clash = [t for t in CONTRADICTION_TERMS
                 if t in head and t in tail.lower()]
        if clash:
            print(f"  WARNING: named as present AND absent: {', '.join(clash)}.")

        # The item-by-item audit. This is the one that acts: what it drops is
        # not sent, so the file no longer has to be fixed by hand before
        # generating - which was the previous instruction, and was never done.
        report_absent(out, text, query_attrs(run), orient)
    C.log(run, f"described garment, {words} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
