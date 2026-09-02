#!/usr/bin/env python3
"""The prompt, as named sections you can edit one at a time.

    python tools/promptfile.py --run runs/<stamp> --show
    python tools/promptfile.py --run runs/<stamp> --set pose --text "..."
    python tools/promptfile.py --run runs/<stamp> --remove reference
    python tools/promptfile.py --run runs/<stamp> --replace-file draft.txt

Stored at `<run>/archive/prompt_sections.json`, rendered to one block of text
whenever generate.py needs it.

WHY SECTIONS. The old pipeline had one prompt, written blind before the first
image was bought and never touched again. When a batch came back with the right
garment in the wrong pose, there was no way to change the pose sentence without
rewriting the whole thing and re-rolling every other decision with it. Sections
make the edit surgical: change `pose`, leave `fidelity` byte-for-byte alone, and
the next generation differs in exactly one respect.

Sections render in a fixed order regardless of the order they were written in,
so an edit never silently reshuffles the prompt. Empty sections render as
nothing, so the seeded skeleton below costs nothing until it is filled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

FILENAME = "prompt_sections.json"

# The slots, in render order, each with the question it answers. Seeded EMPTY on
# purpose: pre-filled boilerplate is a prompt the model can ship without ever
# looking at the garment, which is the failure this whole project exists to fix.
# The hint tells it what belongs there; the words have to be its own.
SEEDS: tuple[tuple[str, str], ...] = (
    ("garment",
     "What the item actually is, read off image 1 - type, colour, closure, "
     "sleeve and hem treatment, any trim. Names the thing so the model does not "
     "invent a different one."),
    ("fidelity",
     "What must survive: the construction exactly as it is - every seam, "
     "pocket, waistband, cuff, label and logo - and the colour and pattern. "
     "Say nothing about keeping creases, folds, texture or proportions: the "
     "lay, the flatness and the size in the frame come from image 2."),
    ("flatten",
     "The job. The garment lies completely flat and smooth, every crease and "
     "handling fold gone, as in a finished catalogue flat. The fabric still "
     "reads as its own fabric - fleece, knit, denim - just pressed."),
    ("pose",
     "How it should sit, read off image 2: square to the frame, hem parallel "
     "to the bottom edge, sleeves or legs at image 2's angle and spacing, no "
     "tilt, and the garment the same size in the frame as image 2's, with "
     "the same margins."),
    ("background",
     "The plate: plain, even white, no shadows, no props, nothing else in "
     "frame."),
    ("reference",
     "What image 2 is for, in your own words - the layout guide. Say it sets "
     "the lay, the flatness and the size in frame, and that its construction "
     "and trim do not belong on this garment."),
)

SEED_ORDER = [name for name, _ in SEEDS]
HINTS = dict(SEEDS)


def path_for(run: Path) -> Path:
    return Path(run) / "archive" / FILENAME


def load(run: Path) -> dict:
    """The stored prompt, seeded on first use.

    A file that exists but cannot be parsed is not silently replaced - the model
    may have hand-edited it and losing its wording without a word is worse than
    an error it can see and fix.
    """
    f = path_for(run)
    if not f.exists():
        return {"order": list(SEED_ORDER),
                "sections": {n: "" for n in SEED_ORDER}}
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"{f} is not readable JSON ({e}). Fix or delete it; "
                           f"deleting reseeds the empty skeleton.") from e
    data.setdefault("order", list(SEED_ORDER))
    data.setdefault("sections", {})
    for n in data["order"]:                      # a name in order but not in
        data["sections"].setdefault(n, "")       # sections renders as nothing
    return data


def save(run: Path, data: dict) -> Path:
    f = path_for(run)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2) + "\n")
    return f


def set_section(run: Path, name: str, text: str) -> str:
    """Add a section or replace one. Unknown names are appended, not refused.

    The six seeded slots are a starting shape, not a schema - a garment with a
    detail none of them covers should get its own section rather than have it
    crammed into `fidelity`.
    """
    name = name.strip().lower().replace(" ", "_")
    if not name:
        return "ERROR: a section needs a name."
    data = load(run)
    had = (data["sections"].get(name) or "").strip()
    if name not in data["order"]:
        data["order"].append(name)
        verb = "added"
    else:
        verb = "replaced" if had else "filled"
    data["sections"][name] = text.strip()
    save(run, data)
    words = len(text.split())
    return (f"{verb} section '{name}' ({words} words). "
            f"Prompt is now {len(render(run).split())} words across "
            f"{len(filled(run))} filled section(s).")


def remove_section(run: Path, name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    data = load(run)
    if name not in data["order"]:
        return (f"ERROR: no section '{name}'. Sections: "
                f"{', '.join(data['order']) or '(none)'}")
    data["order"].remove(name)
    data["sections"].pop(name, None)
    save(run, data)
    return (f"removed section '{name}'. Prompt is now "
            f"{len(render(run).split())} words across {len(filled(run))} "
            f"filled section(s).")


def replace_all(run: Path, text: str) -> str:
    """Throw the sections away and keep one block under `prompt`.

    Deliberately destructive and deliberately available: sometimes the structure
    is the thing that is wrong, and rewriting six sections one call at a time to
    say what one paragraph says is six turns spent on nothing.
    """
    data = {"order": ["prompt"], "sections": {"prompt": text.strip()}}
    save(run, data)
    return (f"replaced the whole prompt with a single section ({len(text.split())} "
            f"words). The seeded sections are gone; prompt_set adds new ones.")


def filled(run: Path) -> list[str]:
    data = load(run)
    return [n for n in data["order"] if (data["sections"].get(n) or "").strip()]


def render(run: Path) -> str:
    """The sections joined into the text that actually gets sent."""
    data = load(run)
    parts = [(data["sections"].get(n) or "").strip() for n in data["order"]]
    return "\n\n".join(p for p in parts if p)


def show(run: Path, reference: Path | None = None) -> str:
    """Everything the model needs to decide whether to spend: the assembled
    prompt, which slots are still empty, and what the guardrails make of it."""
    data = load(run)
    text = render(run)
    lines = [f"prompt: {len(text.split())} words, "
             f"{len(filled(run))}/{len(data['order'])} sections filled",
             ""]

    for n in data["order"]:
        body = (data["sections"].get(n) or "").strip()
        if body:
            lines.append(f"[{n}]  {len(body.split())} words")
            lines.append(f"  {body}")
        else:
            lines.append(f"[{n}]  EMPTY - {HINTS.get(n, 'your own section')}")
        lines.append("")

    problems, warnings = C.check_prompt(text, reference)
    lines.append("guardrails:")
    lines.append(C.format_findings(problems, warnings))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path,
                    help="checked for greyscale when showing the guardrails")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--render", action="store_true", help="the raw prompt text")
    ap.add_argument("--set", metavar="SECTION")
    ap.add_argument("--text", help="body for --set; omit to read stdin")
    ap.add_argument("--remove", metavar="SECTION")
    ap.add_argument("--replace-file", type=Path,
                    help="replace the whole prompt with this file's contents")
    a = ap.parse_args()

    ref = a.reference
    if ref is None:
        guess = C.reference_path(a.run)
        ref = guess if guess.exists() else None

    if a.set:
        text = a.text if a.text is not None else sys.stdin.read()
        print(set_section(a.run, a.set, text))
    if a.remove:
        print(remove_section(a.run, a.remove))
    if a.replace_file:
        print(replace_all(a.run, a.replace_file.read_text()))
    if a.render:
        print(render(a.run))
    if a.show or not any((a.set, a.remove, a.replace_file, a.render)):
        print(show(a.run, ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
