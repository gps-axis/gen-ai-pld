#!/usr/bin/env python3
"""Build an HTML contact sheet of laydown runs.

Reads every run folder under runs/ and writes runs/contact_sheet.html:
per run a hero strip (input / segmented / reference / best) plus every
generation with its seed, prompt hash and the agent's verdict. Thumbnails
are written to runs/_thumbs/ so the page stays light; every tile links to
the full-resolution original.

The raw input photo is NOT archived per run - every run segments
`inputs/off_set_image.jpg`, which the next run overwrites. The originals
do survive in inputs/ under their own names, so each run's input is
recovered by matching its archived `source_clean.jpg` against every file
in inputs/ over the garment mask (segmentation keeps garment pixels
intact, so the true source scores near zero and everything else is an
order of magnitude worse). Matches are labelled as recovered, and a run
whose match is not decisive shows no input tile rather than a guess.

Usage:  python3 tools/make_contact_sheet.py [runs_dir]
"""

import html
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

THUMB_DIR_NAME = "_thumbs"
CAND_MAX = 560
HERO_MAX = 900

# Input recovery: a match counts only if it is both close in absolute terms
# and clearly ahead of the next best file. Measured spread on this set is
# 0.5-4.3 for the true source against 23+ for every runner-up.
MATCH_SIZE = (180, 240)
MATCH_MAX_DIFF = 12.0
MATCH_MIN_RATIO = 3.0


# ---------------------------------------------------------------- helpers

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def rel_to(src: Path, runs_dir: Path) -> str:
    """Link target relative to the page, which sits in runs_dir."""
    return os.path.relpath(src, runs_dir)


def thumb(src: Path, runs_dir: Path, max_edge: int) -> str:
    """Write a JPEG thumbnail and return its path relative to runs_dir."""
    if runs_dir in src.parents:
        rel = src.relative_to(runs_dir)
        out = runs_dir / THUMB_DIR_NAME / rel.parent / f"{rel.stem}_{max_edge}.jpg"
    else:                                   # raw inputs live outside runs/
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", src.stem)
        out = runs_dir / THUMB_DIR_NAME / "_inputs" / f"{stem}_{max_edge}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        im = Image.open(src)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        im.save(out, "JPEG", quality=82, optimize=True)
    return str(out.relative_to(runs_dir))


def dims(path: Path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def parse_steps(path: Path):
    """Return (steps, total_credits, shipped_note) from steps.log."""
    steps, total, shipped = [], None, None
    if not path.exists():
        return steps, total, shipped
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[(\d+)\]\s+(\d\d:\d\d:\d\d)\s+(.*)", line)
        if not m:
            continue
        body = m.group(3)
        tot = re.search(r"total\s+([\d.]+)c\s*$", body)
        if tot:
            total = float(tot.group(1))
            body = body[: tot.start()].strip()
        body = re.sub(r"\s+[\d.]+c$", "", body).strip()
        steps.append((m.group(1), m.group(2), body))
        if body.startswith("shipped"):
            shipped = body
    return steps, total, shipped


def parse_log(path: Path):
    """Return (title, brief) from LOG.md."""
    if not path.exists():
        return None, None
    lines = path.read_text().splitlines()
    title = None
    for ln in lines:
        if ln.startswith("# "):
            title = ln[2:].strip()
            break
    brief, grabbing = [], False
    for ln in lines:
        if ln.startswith("## "):
            head = ln[3:].strip().lower()
            if grabbing:
                break
            grabbing = head in ("goal", "brief", "task")
            continue
        if grabbing and ln.strip():
            brief.append(ln.strip())
    return title, " ".join(brief) if brief else None


def log_metrics(path: Path):
    """Pull per-candidate metrics out of a markdown table in LOG.md."""
    out = {}
    if not path.exists():
        return out
    header = None
    for ln in path.read_text().splitlines():
        if not ln.strip().startswith("|"):
            header = None if header and not ln.strip() else header
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        low = [c.lower() for c in cells]
        if low and low[0] in ("cand", "candidate"):
            header = low
            continue
        if not header:
            continue
        key = cells[0].strip("*` ")
        if re.fullmatch(r"\d+", key):
            key = f"cand_{int(key):02d}"
        if not key.startswith("cand_"):
            continue
        row = {}
        for name, val in zip(header[1:], cells[1:]):
            if name in ("iou", "de", "dE".lower(), "wrinkle", "specks"):
                row[name] = val.strip("*` ")
        out[key] = row
    return out


def input_bank(inputs_dir: Path):
    """Downscaled RGB arrays for every candidate raw input."""
    bank = {}
    if not inputs_dir.is_dir():
        return bank
    for p in sorted(inputs_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if "reference" in p.name.lower():          # lay guides, not sources
            continue
        try:
            with Image.open(p) as im:
                bank[p] = np.asarray(im.convert("RGB").resize(MATCH_SIZE, Image.LANCZOS),
                                     dtype=np.float32)
        except Exception:
            pass
    return bank


def match_input(seg: Path, bank):
    """Find the raw photo `seg` was segmented from.

    Compares only the garment pixels (everything the segmenter did not turn
    white), so the backdrop it removed cannot drag the score around.
    Returns (path, diff, runner_up_diff) or None when the call is not clear.
    """
    if not bank or seg is None or not seg.exists():
        return None
    with Image.open(seg) as im:
        a = np.asarray(im.convert("RGB").resize(MATCH_SIZE, Image.LANCZOS), dtype=np.float32)
    mask = a.min(axis=2) < 235
    if mask.sum() < 200:
        return None
    scored = sorted((np.abs(a - b).mean(axis=2)[mask].mean(), p) for p, b in bank.items())
    best, second = scored[0], (scored[1] if len(scored) > 1 else (float("inf"), None))
    if best[0] > MATCH_MAX_DIFF or second[0] < best[0] * MATCH_MIN_RATIO:
        return None
    return best[1], best[0], second[0]


def run_times(run_id, steps):
    """(started, last_step, span) - start from the folder name, end from the
    last line of steps.log. Both are naive local times; None when unknown."""
    try:
        started = datetime.strptime(run_id, "%Y%m%d_%H%M%S")
    except ValueError:
        return None, None, None
    if not steps:
        return started, None, None
    hh, mm, ss = (int(x) for x in steps[-1][1].split(":"))
    ended = started.replace(hour=hh, minute=mm, second=ss)
    if ended < started:                     # ran across midnight
        ended += timedelta(days=1)
    return started, ended, ended - started


def short(text, n):
    """Cut to n characters at a word boundary."""
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut or text[:n]).rstrip(" ,-") + "\u2026"


def fmt_span(td):
    if td is None:
        return ""
    secs = int(td.total_seconds())
    m, s = divmod(secs, 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def title_from_brief(brief, run_id, sections=None, selection=None):
    if brief:
        m = re.search(r"(?:Re-lay|Relay)\s+(?:the\s+)?(.+?)(?=\s+into\s|\s*[(,.:;]|$)", brief)
        if m:
            return m.group(1).strip().rstrip(".")
        return brief.split(".")[0][:70]
    # No LOG.md yet (run still going, or cut short): name it from the prompt's
    # garment section, first clause, leading article dropped.
    garment = ((sections or {}).get("sections") or {}).get("garment", "")
    first = re.split(r"[,.;:]|\s+with\s+", garment, maxsplit=1)[0].strip()
    first = re.sub(r"^(?:an?|the)\s+", "", first, flags=re.I)
    if first:
        return first[:70]
    # Or from what the reference selector saw, for a run that stopped there.
    attrs = (selection or {}).get("query_attrs") or {}
    desc = " ".join(x for x in (attrs.get("color_name"), attrs.get("garment_type")) if x).strip()
    return desc[:70] if desc else run_id


# ---------------------------------------------------------------- run model

def collect_run(run_dir: Path, runs_dir: Path, bank=None):
    arch = run_dir / "archive"
    if not arch.is_dir():
        return None

    run_id = run_dir.name
    best = load_json(arch / "best.json")
    notes = load_json(arch / "notes.json")
    lineage = load_json(arch / "lineage.json")
    seeds = load_json(arch / "seeds.json")
    sections = load_json(arch / "prompt_sections.json")
    polish = load_json(arch / "last_polish.json")
    picks_doc = load_json(run_dir / "output" / "picks.json")
    selection = load_json(run_dir / "reference_selection.json")
    steps, total_c, shipped = parse_steps(run_dir / "steps.log")
    log_title, brief = parse_log(run_dir / "LOG.md")
    metrics = log_metrics(run_dir / "LOG.md")

    winner = best.get("candidate")
    polished_child = polish.get("child")
    shipped_polish = bool(shipped and polished_child and polished_child in shipped)

    cands = []
    for png in sorted(arch.glob("cand_*.png")):
        name = png.stem
        # Derived names carry a suffix per step: s segmented, p polished,
        # c recoloured. The parent is the name with the last letter dropped.
        is_polish = name.endswith("p")
        derived = bool(re.fullmatch(r"cand_\d+[spc]+", name))
        parent = name[:-1] if derived else None
        info = lineage.get(name, {}) or seeds.get(name, {})
        cands.append({
            "name": name,
            "path": png,
            "is_polish": is_polish,
            "parent": parent,
            "from": info.get("parent") or info.get("source"),
            "is_winner": name == winner,
            "seed": info.get("seed"),
            "prompt": info.get("prompt_hash") or info.get("prompt"),
            "res": info.get("resolution"),
            "note": notes.get(name),
            "metrics": metrics.get(name, {}),
        })
    # polished children sit next to their parent
    order = {c["name"]: i for i, c in enumerate(c for c in cands if not c["is_polish"])}
    cands.sort(key=lambda c: (order.get(c["parent"] or c["name"], 99), c["is_polish"]))

    seg = arch / "source_clean.jpg" if (arch / "source_clean.jpg").exists() else None
    m = match_input(seg, bank)

    # The harness ships a ranked set: best.png, best_2.png, ... with the
    # measurements it ranked on. Keep only picks whose file is really there.
    picks = []
    for pk in picks_doc.get("picks", []) or []:
        f = run_dir / "output" / str(pk.get("file", ""))
        if pk.get("file") and f.exists():
            picks.append({**pk, "path": f})

    started, ended, span = run_times(run_id, steps)

    # A run that never generated stopped for a reason the log names; the
    # reference selector's refusal is the one seen so far.
    stopped = None
    for _, _, body in steps:
        if body.startswith("reference NOT selected"):
            m_stop = re.search(r"\((.*?)\)", body)
            stopped = "no reference match" + (f" ({m_stop.group(1)})" if m_stop else "")
    ref_sheet = run_dir / "reference_match.jpg"

    return {
        "id": run_id,
        "dir": run_dir,
        "started": started,
        "ended": ended,
        "span": span,
        "title": title_from_brief(brief, run_id, sections, selection),
        "log_title": log_title,
        "brief": brief,
        "input": m[0] if m else None,
        "input_score": (m[1], m[2]) if m else None,
        "segmented": seg,
        "reference": ref if (ref := C.reference_path(run_dir)).exists() else None,
        "ref_sheet": ref_sheet if ref_sheet.exists() else None,
        "stopped": stopped,
        "best_png": run_dir / "output" / "best.png" if (run_dir / "output" / "best.png").exists() else None,
        "winner": winner,
        "why": best.get("why"),
        "override": best.get("override") or picks_doc.get("override"),
        "cands": cands,
        "cands_notes": notes,
        "total_c": total_c,
        "steps": steps,
        "shipped": shipped,
        "shipped_polish": shipped_polish,
        "polish": polish,
        "picks": picks,
        "picks_note": picks_doc.get("note"),
        "sections": sections,
        "complete": bool(winner and (run_dir / "output" / "best.png").exists()),
    }


# ---------------------------------------------------------------- rendering

def esc(x):
    return html.escape(str(x)) if x is not None else ""


def figure(src: Path, runs_dir: Path, max_edge, label, sub="", cls="", badges=(), note=None):
    if src is None or not src.exists():
        return (f'<figure class="tile {cls} missing"><div class="ph">no image</div>'
                f'<figcaption><span class="lbl">{esc(label)}</span></figcaption></figure>')
    t = thumb(src, runs_dir, max_edge)
    full = rel_to(src, runs_dir)
    wh = dims(src)
    size = f"{wh[0]}x{wh[1]}" if wh else ""
    badge_html = "".join(f'<span class="badge {b[1]}">{esc(b[0])}</span>' for b in badges)
    note_html = f'<p class="note">{esc(note)}</p>' if note else ""
    return f"""<figure class="tile {cls}">
  <a href="{esc(full)}" class="shot" data-full="{esc(full)}" data-title="{esc(label)}">
    <img src="{esc(t)}" alt="{esc(label)}" loading="lazy">
    {f'<span class="badges">{badge_html}</span>' if badge_html else ''}
  </a>
  <figcaption>
    <span class="lbl">{esc(label)}</span>
    <span class="sub">{esc(sub)}{(' &middot; ' + size) if sub and size else esc(size)}</span>
    {note_html}
  </figcaption>
</figure>"""


def render_run(run, runs_dir):
    r = run
    parts = []
    n_gen = sum(1 for c in r["cands"] if not c["is_polish"])
    n_pol = sum(1 for c in r["cands"] if c["is_polish"])

    chips = [f'<span class="chip">{n_gen} generations</span>']
    if n_pol:
        chips.append(f'<span class="chip">{n_pol} polish</span>')
    if r["total_c"] is not None:
        chips.append(f'<span class="chip">{r["total_c"]:.0f} credits</span>')
    if r["span"] is not None:
        chips.append(f'<span class="chip">{fmt_span(r["span"])}</span>')
    if r["winner"]:
        chips.append(f'<span class="chip win">winner {esc(r["winner"])}</span>')
    if len(r["picks"]) > 1:
        chips.append(f'<span class="chip">{len(r["picks"])} shipped, ranked</span>')
    if r["stopped"]:
        chips.append(f'<span class="chip warn">{esc(r["stopped"])}</span>')
    if not r["complete"]:
        chips.append('<span class="chip warn">no ship</span>')

    when = ""
    if r["started"]:
        when = r["started"].strftime("%Y-%m-%d %H:%M:%S")
        if r["ended"]:
            tail = "shipped" if r["shipped"] else "last step"
            when += f" \u2192 {r['ended'].strftime('%H:%M:%S')} {tail}"
            if r["span"] is not None:
                when += f" \u00b7 {fmt_span(r['span'])}"

    parts.append(f"""<section class="run" id="{esc(r['id'])}">
  <header class="runhead">
    <div>
      <h2>{esc(r['title'])}</h2>
      <div class="runid">{esc(r['id'])}{(' &middot; ' + esc(r['log_title'])) if r['log_title'] else ''}</div>
      <div class="runtime">{esc(when)}</div>
    </div>
    <div class="chips">{''.join(chips)}</div>
  </header>""")

    if r["brief"]:
        parts.append(f'<p class="brief">{esc(r["brief"])}</p>')

    # hero strip: the two things that went in, what the agent worked from, what shipped
    if r["input"]:
        d, second = r["input_score"]
        in_sub = f"{r['input'].name} · recovered, diff {d:.1f} vs {second:.0f}"
    else:
        in_sub = "not archived, no confident match"
    hero = [
        figure(r["input"], runs_dir, HERO_MAX, "Input photo", in_sub, "hero",
               [("input", "grey")]),
        figure(r["segmented"], runs_dir, HERO_MAX, "Segmented", "background dropped, garment untouched",
               "hero", [("step 0", "grey")]),
        (figure(r["reference"], runs_dir, HERO_MAX, "Reference", "lay guide only", "hero",
                [("ref", "grey")])
         if r["reference"] or not r["ref_sheet"] else
         figure(r["ref_sheet"], runs_dir, HERO_MAX, "Reference",
                (r["stopped"] or "not selected") + " - the selector's own sheet of the library",
                "hero", [("no ref", "grey")])),
    ]
    best_sub = "shipped"
    if r["shipped_polish"]:
        best_sub = f"polished from {esc(r['winner'])}"
    elif r["winner"]:
        best_sub = f"{esc(r['winner'])} shipped as-is"
    hero.append(figure(r["best_png"], runs_dir, HERO_MAX, "Best", best_sub, "hero best",
                       [("best", "gold")]))
    parts.append(f'<div class="hero-row">{"".join(hero)}</div>')

    ov = r["override"]
    if ov:
        parts.append(f'<div class="why override"><span class="why-k">Operator override</span>'
                     f'<p>{esc(re.sub(r"^operator override:\s*", "", ov.get("note") or ""))}{(" \u00b7 " + esc(ov["at"][:16].replace("T", " "))) if ov.get("at") else ""}'
                     f' \u00b7 the model had picked {esc(ov.get("was", ""))}; its reasoning is kept below.</p></div>')
    if r["why"]:
        parts.append(f'<div class="why"><span class="why-k">Why this one</span>'
                     f'<p>{esc(r["why"])}</p></div>')

    # shipped picks, in the harness's rank order, with what it ranked on
    if r["picks"]:
        tiles = []
        for pk in r["picks"]:
            rank = pk.get("rank")
            meta = []
            if pk.get("lay_iou") is not None:
                meta.append(f"lay IoU {pk['lay_iou']:.3f}")
            if pk.get("size_vs_ref") is not None:
                meta.append(f"size x{pk['size_vs_ref']:.2f}")
            if pk.get("wrinkle_ratio") is not None:
                meta.append(f"wrinkle {pk['wrinkle_ratio']:.2f}")
            if pk.get("colour_de_lit") is not None:
                hue = f" (hue {pk['hue_de']:.1f})" if pk.get("hue_de") is not None else ""
                meta.append(f"dE {pk['colour_de_lit']:.1f}{hue}")
            if pk.get("silhouette_iou") is not None:
                meta.append(f"silhouette IoU {pk['silhouette_iou']:.3f}")
            badge = [(f"#{rank}", "gold" if rank == 1 else "grey")]
            label = f"{pk.get('candidate', '')} -> {pk.get('file', '')}"
            cls = "cand" + (" is-winner" if rank == 1 else "")
            tiles.append(figure(pk["path"], runs_dir, CAND_MAX, label, " · ".join(meta), cls,
                                badge, r["cands_notes"].get(pk.get("candidate"))))
        note = f'<p class="brief">{esc(r["picks_note"])}</p>' if r["picks_note"] else ""
        parts.append(f'<h3 class="sec">Shipped, ranked</h3><div class="grid picks">{"".join(tiles)}</div>{note}')

    # generations grid
    tiles = []
    for c in r["cands"]:
        badges = []
        if c["is_winner"]:
            badges.append(("selected", "gold"))
        if c["is_polish"]:
            badges.append(("polish", "blue"))
        meta = []
        if c["from"] and c["from"] != "source":
            meta.append(f"from {c['from']}")
        if c["seed"] is not None:
            meta.append(f"seed {c['seed']}")
        if c["prompt"]:
            meta.append(f"prompt {c['prompt']}")
        if c["res"]:
            meta.append(str(c["res"]))
        m = c["metrics"]
        if m:
            mm = [f"{k} {v}" for k, v in m.items() if v]
            if mm:
                meta.append(" / ".join(mm))
        if c["is_polish"] and r["polish"].get("child") == c["name"]:
            p = r["polish"]
            meta.append(f"IoU {p.get('silhouette_iou')} / dE {p.get('colour_de')}")
        cls = "cand" + (" is-winner" if c["is_winner"] else "") + (" is-polish" if c["is_polish"] else "")
        tiles.append(figure(c["path"], runs_dir, CAND_MAX, c["name"], " · ".join(meta), cls,
                            badges, c["note"]))
    parts.append(f'<h3 class="sec">All generations</h3><div class="grid">{"".join(tiles)}</div>')

    # details
    det = []
    secs = r["sections"].get("sections") if r["sections"] else None
    if secs:
        order = r["sections"].get("order") or list(secs)
        rows = "".join(f'<div class="prow"><span class="pk">{esc(k)}</span>'
                       f'<span class="pv">{esc(secs.get(k, ""))}</span></div>'
                       for k in order if secs.get(k))
        det.append(f'<details><summary>Prompt sections</summary><div class="prompt">{rows}</div></details>')
    if r["steps"]:
        rows = "".join(f'<div class="srow"><span class="sn">{esc(n)}</span>'
                       f'<span class="st">{esc(t)}</span><span class="sb">{esc(b)}</span></div>'
                       for n, t, b in r["steps"])
        det.append(f'<details><summary>Run steps</summary><div class="steps">{rows}</div></details>')
    if det:
        parts.append(f'<div class="details">{"".join(det)}</div>')

    parts.append("</section>")
    return "\n".join(parts)


CSS = """
:root{--bg:#0d0f12;--panel:#15181d;--panel2:#1b1f26;--line:#262b33;--tx:#e8eaed;
--dim:#9aa2ad;--faint:#6b7280;--gold:#f0b429;--blue:#5aa9e6;--warn:#e06c5a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
a{color:inherit;text-decoration:none}
header.top{position:sticky;top:0;z-index:20;background:rgba(13,15,18,.94);
backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 28px}
header.top h1{margin:0;font-size:17px;letter-spacing:.2px}
header.top .meta{color:var(--dim);font-size:12px;margin-top:3px}
nav.runs{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
nav.runs a{font-size:11px;color:var(--dim);border:1px solid var(--line);
border-radius:999px;padding:3px 9px;background:var(--panel)}
nav.runs a:hover{color:var(--tx);border-color:#3a424e}
main{padding:22px 28px 80px;max-width:1800px;margin:0 auto}
section.run{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:20px 22px 24px;margin-bottom:26px}
.runhead{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;
flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:12px}
.runhead h2{margin:0;font-size:19px;font-weight:600}
.runid{color:var(--faint);font-size:12px;margin-top:3px;font-family:ui-monospace,Menlo,monospace}
.runtime{color:var(--dim);font-size:12px;margin-top:2px;font-family:ui-monospace,Menlo,monospace}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:11px;color:var(--dim);background:var(--panel2);border:1px solid var(--line);
border-radius:999px;padding:3px 10px;white-space:nowrap}
.chip.win{color:var(--gold);border-color:#5a4a1e}
.chip.warn{color:var(--warn);border-color:#5a2f28}
.brief{color:var(--dim);margin:14px 0 0;max-width:110ch;font-size:13px}
.hero-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:16px}
.note-top{max-width:120ch;line-height:1.55;margin-top:6px}
.note-top code{background:var(--panel2);border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font-size:11px}
h3.sec{font-size:12px;text-transform:uppercase;letter-spacing:.9px;color:var(--faint);
margin:26px 0 10px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.grid.picks{grid-template-columns:repeat(auto-fill,minmax(250px,1fr))}
.tile{margin:0;background:var(--panel2);border:1px solid var(--line);border-radius:9px;
overflow:hidden;display:flex;flex-direction:column}
.tile .shot{position:relative;display:block;background:#fff;aspect-ratio:3/4}
.tile.hero .shot{aspect-ratio:3/4}
.tile img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block}
.tile.is-winner{border-color:var(--gold);box-shadow:0 0 0 1px rgba(240,180,41,.35)}
.tile.best{border-color:var(--gold)}
.tile.is-polish{border-color:#2c4a63}
.badges{position:absolute;top:7px;left:7px;display:flex;gap:4px}
.badge{font-size:10px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;
padding:2px 7px;border-radius:4px;background:#111;color:#ddd}
.badge.gold{background:var(--gold);color:#1a1200}
.badge.blue{background:var(--blue);color:#04121f}
.badge.grey{background:rgba(0,0,0,.6);color:#e5e7eb}
figcaption{padding:8px 10px 10px;border-top:1px solid var(--line)}
.lbl{display:block;font-size:12px;font-weight:600;font-family:ui-monospace,Menlo,monospace}
.sub{display:block;font-size:11px;color:var(--faint);margin-top:2px;
font-family:ui-monospace,Menlo,monospace;word-break:break-word}
.note{margin:7px 0 0;font-size:11.5px;color:var(--dim);line-height:1.45}
.missing .ph{aspect-ratio:3/4;display:grid;place-items:center;color:var(--faint);
font-size:12px;background:repeating-linear-gradient(45deg,#15181d,#15181d 8px,#191d24 8px,#191d24 16px)}
.why{margin-top:14px;background:rgba(240,180,41,.07);border:1px solid #4a3d18;
border-left:3px solid var(--gold);border-radius:7px;padding:11px 14px}
.why.override{background:rgba(224,108,90,.08);border-color:#5a2f28;border-left-color:var(--warn)}
.why.override .why-k{color:var(--warn)}
.why-k{font-size:10.5px;text-transform:uppercase;letter-spacing:.9px;color:var(--gold);font-weight:600}
.why p{margin:5px 0 0;color:#d9dde3;font-size:13px;max-width:120ch}
.details{margin-top:20px;display:flex;flex-direction:column;gap:8px}
details{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:9px 13px}
summary{cursor:pointer;font-size:12px;color:var(--dim);user-select:none}
summary:hover{color:var(--tx)}
.prompt,.steps{margin-top:10px;display:flex;flex-direction:column;gap:6px}
.prow{display:grid;grid-template-columns:110px 1fr;gap:12px;font-size:12px;
padding-bottom:6px;border-bottom:1px solid #22262e}
.pk{color:var(--blue);font-family:ui-monospace,Menlo,monospace}
.pv{color:var(--dim)}
.srow{display:grid;grid-template-columns:28px 74px 1fr;gap:10px;font-size:12px;
font-family:ui-monospace,Menlo,monospace;color:var(--dim)}
.sn{color:var(--faint)}
.st{color:var(--faint)}
#lb{position:fixed;inset:0;background:rgba(6,7,9,.94);display:none;z-index:50;
align-items:center;justify-content:center;flex-direction:column;gap:12px;padding:24px}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:86vh;object-fit:contain;background:#fff;border-radius:6px}
#lb .cap{color:var(--dim);font-size:12px;font-family:ui-monospace,Menlo,monospace}
@media(max-width:900px){.hero-row{grid-template-columns:1fr}main{padding:16px}}
"""

JS = """
const lb=document.getElementById('lb'),lbi=lb.querySelector('img'),lbc=lb.querySelector('.cap');
document.querySelectorAll('a.shot').forEach(a=>a.addEventListener('click',e=>{
  e.preventDefault();lbi.src=a.dataset.full;lbc.textContent=a.dataset.title+'  -  '+a.dataset.full;
  lb.classList.add('on');}));
lb.addEventListener('click',()=>{lb.classList.remove('on');lbi.src='';});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){lb.classList.remove('on');lbi.src='';}});
"""


def build(runs_dir: Path, inputs_dir: Path):
    bank = input_bank(inputs_dir)
    runs = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}_\d{6}", p.name)):
        r = collect_run(d, runs_dir, bank)
        if r:
            runs.append(r)

    total_gen = sum(sum(1 for c in r["cands"] if not c["is_polish"]) for r in runs)
    total_c = sum(r["total_c"] or 0 for r in runs)
    shipped = sum(1 for r in runs if r["complete"])
    unmatched = sum(1 for r in runs if not r["input"])

    nav = "".join(
        f'<a href="#{esc(r["id"])}">'
        f'{esc(r["started"].strftime("%H:%M") if r["started"] else r["id"][9:])} {esc(short(r["title"], 28))}</a>'
        for r in runs)
    today = date.today().isoformat()
    body = "\n".join(render_run(r, runs_dir) for r in runs)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laydown agent runs - contact sheet - {today}</title>
<style>{CSS}</style></head>
<body>
<header class="top">
  <h1>Laydown agent runs &middot; {today}</h1>
  <div class="meta">built {today} &middot; {len(runs)} runs &middot; {total_gen} generations &middot; {shipped} shipped
   &middot; {total_c:.0f} credits total &middot; click any frame for full resolution</div>
  <div class="meta note-top">Input photos are not archived per run - the harness segments
   <code>inputs/off_set_image.jpg</code> and the next run overwrites it. Each input tile below was
   recovered by matching the run's segmented source against every file in <code>inputs/</code> over
   the garment pixels; the caption carries the match distance and the runner-up.
   {f'{unmatched} run(s) had no confident match and show no input.' if unmatched else 'All runs matched decisively.'}</div>
  <nav class="runs">{nav}</nav>
</header>
<main>
{body}
</main>
<div id="lb"><img alt=""><div class="cap"></div></div>
<script>{JS}</script>
</body></html>
"""
    out = runs_dir / "contact_sheet.html"
    out.write_text(doc)
    return out, len(runs), total_gen


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    base = (Path(sys.argv[1]) if len(sys.argv) > 1 else root / "runs").resolve()
    inputs = (Path(sys.argv[2]) if len(sys.argv) > 2 else root / "inputs").resolve()
    path, n, g = build(base, inputs)
    print(f"wrote {path}  ({n} runs, {g} generations)")
