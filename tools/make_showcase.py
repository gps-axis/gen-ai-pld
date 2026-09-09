#!/usr/bin/env python3
"""One row per run - the original and its four picks - -> runs/showcase.html.

Left to right: the off-set photo that went in, the best pick right beside it
for the direct before-and-after, then the second, third and fourth. Each pick
tile names the candidate it was and who chose it (the model, the judge, or the
harness filling an empty slot), read from output/picks.json.

Every row is also rendered as one PNG under runs/_strips/ - the same five
tiles, tags and badges - behind a "download strip" button, with a zip of all of
them behind the header's button. They are drawn here at build time rather than
in the browser: the page is opened from disk, and a canvas that has drawn a
file:// image cannot be exported.

Reuses the contact sheet's thumbnailer and its input recovery (runs do not
archive the raw photo, so it is matched back from inputs/).

    python3 tools/make_showcase.py              # every run that shipped a best.png
    python3 tools/make_showcase.py [run_id ...] # just these, in this order

Without arguments the set is read off disk rather than kept in this file: a
list written here went stale the first time a batch of runs was archived away,
and a showcase of runs that no longer exist is worse than none.
"""

import json
import re
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_contact_sheet import HERO_MAX, esc, input_bank, match_input, rel_to, thumb

STRIP_DIR_NAME = "_strips"
TILE_W, TILE_H = 600, 800          # 3:4, like the tiles on the page
DIVIDER = 2
CAPTION_H = 72                     # two lines: run / note / source, then the models
GOLD = (240, 180, 41)
INK = (22, 25, 30)
PANEL = (22, 25, 30)
DIM = (139, 147, 158)
LINE = (38, 43, 51)
FILLED = (160, 60, 30)

# The one seeded endpoint this project has used. A lineage entry carrying an
# integer seed can only have come from it; gpt-image has no seed to record.
NANO_BANANA = "fal-ai/nano-banana-pro/edit"


def models_of(run: Path) -> tuple[str, str, str]:
    """(image endpoint, agent model, how the endpoint was known) for one run.

    generate.py writes the endpoint into lineage.json since 2026-09-08. Before
    that nothing in a run folder named it, and two things still pin it down:
    generate.py prints "seed ignored: <endpoint> has none" for a model without
    seeds, and a candidate WITH a seed can only be nano-banana. A run with
    neither is reported as unknown rather than guessed.
    """
    arch = run / "archive"
    try:
        lineage = json.loads((arch / "lineage.json").read_text())
    except (OSError, json.JSONDecodeError):
        lineage = {}
    entries = [v for v in lineage.values() if isinstance(v, dict)]
    endpoint, basis = "", ""
    named = sorted({v["endpoint"] for v in entries if v.get("endpoint")})
    if named:
        endpoint, basis = ", ".join(named), "recorded in lineage.json"
    else:
        log = run / "run.log"
        m = (re.search(r"seed ignored: (\S+) has none", log.read_text(errors="replace"))
             if log.exists() else None)
        if m:
            endpoint, basis = m.group(1), "named in run.log"
        elif any(isinstance(v.get("seed"), int) for v in entries):
            endpoint, basis = NANO_BANANA, "inferred: the candidates carry seeds"
    agent = ""
    t = run / "transcript.jsonl"
    if t.exists():
        with t.open() as f:
            first = f.readline()
        try:
            rec = json.loads(first)
            if rec.get("kind") == "start":
                agent = str(rec.get("data", {}).get("model") or "")
        except json.JSONDecodeError:
            pass
    return endpoint, agent, basis


def short_model(endpoint: str) -> str:
    """'openai/gpt-image-2.5/sunburst/edit' -> 'gpt-image-2.5/sunburst'."""
    return re.sub(r"/edit$", "", re.sub(r"^(fal-ai|openai)/", "", endpoint))


def shipped_runs(runs_dir: Path) -> list[str]:
    """Every run folder that delivered a best.png, oldest first."""
    return sorted(d.name for d in runs_dir.iterdir()
                  if d.is_dir() and (d / "output" / "best.png").exists())

CSS = """
:root{--bg:#0e0f12;--panel:#16191e;--line:#262b33;--tx:#e8eaed;--dim:#8b939e;--gold:#f0b429}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{margin:0;font-size:18px;font-weight:600}
header{padding:20px 26px 16px;border-bottom:1px solid var(--line)}
header p{margin:4px 0 0;color:var(--dim);font-size:12.5px}
main{padding:22px 26px 60px;display:grid;gap:20px;max-width:1700px;margin:0 auto}
.pair{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.imgs{display:grid;grid-template-columns:repeat(5,1fr)}
.imgs a{display:block;background:#fff;aspect-ratio:3/4;position:relative;border-right:1px solid var(--line)}
.imgs a:last-child{border-right:0}
.imgs a.empty{background:var(--panel)}
.imgs a.best{box-shadow:inset 0 0 0 3px var(--gold)}
.imgs img{width:100%;height:100%;object-fit:contain;display:block}
.tag{position:absolute;top:8px;left:8px;font-size:10px;font-weight:600;letter-spacing:.5px;
text-transform:uppercase;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.62);color:#eef1f4}
.tag.out{background:var(--gold);color:#1a1200}
.who{position:absolute;right:8px;bottom:8px;font-size:10px;letter-spacing:.3px;padding:2px 7px;
border-radius:4px;background:rgba(0,0,0,.62);color:#eef1f4;font-family:ui-monospace,Menlo,monospace}
.who.filled{background:rgba(160,60,30,.85)}
.cap{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 12px;
border-top:1px solid var(--line);font-size:11.5px;color:var(--dim);
font-family:ui-monospace,Menlo,monospace;flex-wrap:wrap}
.cap .models{flex-basis:100%;cursor:help}
.dl{color:var(--dim);text-decoration:underline;text-underline-offset:3px;white-space:nowrap}
.dl:hover{color:var(--tx)}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:16px 26px;border-bottom:1px solid var(--line)}
header .dl{color:#1a1200;background:var(--gold);text-decoration:none;font-weight:600;font-size:12px;padding:7px 14px;border-radius:5px;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
header .dl:hover{filter:brightness(1.08);color:#1a1200}
@media(max-width:900px){.imgs{grid-template-columns:repeat(5,minmax(120px,1fr))}.pair{overflow-x:auto}}
#lb{position:fixed;inset:0;background:rgba(6,7,9,.95);display:none;z-index:50;
align-items:center;justify-content:center;padding:24px}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:94vh;object-fit:contain;background:#fff;border-radius:6px}
@media(max-width:520px){main{padding:16px}}
"""

JS = """
const lb=document.getElementById('lb'),i=lb.querySelector('img');
document.querySelectorAll('main a:not(.dl)').forEach(a=>a.addEventListener('click',e=>{
  e.preventDefault();i.src=a.getAttribute('href');lb.classList.add('on');}));
lb.addEventListener('click',()=>{lb.classList.remove('on');i.src='';});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){lb.classList.remove('on');i.src='';}});
"""


def shot(src: Path, runs_dir: Path, tag: str, cls: str, who: str = "",
         who_cls: str = "", frame: str = "") -> str:
    if src is None or not src.exists():
        return f'<a class="empty {frame}"><span class="tag {cls}">{esc(tag)}</span></a>'
    badge = f'<span class="who {who_cls}">{esc(who)}</span>' if who else ""
    return (f'<a class="{frame}" href="{esc(rel_to(src, runs_dir))}">'
            f'<img src="{esc(thumb(src, runs_dir, HERO_MAX))}" alt="{esc(tag)}" loading="lazy">'
            f'<span class="tag {cls}">{esc(tag)}</span>{badge}</a>')


RANK_TAG = {1: "best", 2: "2nd", 3: "3rd", 4: "4th"}


def picks_by_rank(run: Path) -> dict:
    """rank -> pick record from output/picks.json; empty when the run has none."""
    f = run / "output" / "picks.json"
    if not f.exists():
        return {}
    try:
        return {int(r["rank"]): r for r in json.loads(f.read_text()).get("picks", [])
                if isinstance(r, dict) and "rank" in r}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def pick_tile(run: Path, runs_dir: Path, rank: int, picks: dict) -> str:
    rec = picks.get(rank, {})
    name = rec.get("file") or ("best.png" if rank == 1 else f"best_{rank}.png")
    src = run / "output" / name
    who = rec.get("chosen_by", "")
    label = (f"{rec.get('candidate', '')} {who}".strip()) if rec else ""
    return shot(src, runs_dir, RANK_TAG[rank], "out" if rank == 1 else "",
                who=label, who_cls="filled" if who == "harness" else "",
                frame="best" if rank == 1 else "")


def _font(size: int, bold: bool = False):
    candidates = (
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
         else "/System/Library/Fonts/Supplemental/Arial.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
         else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    )
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def _mono(size: int):
    for path in ("/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pill(draw, xy, text, font, bg, fg, pad=(10, 5), anchor="lt"):
    """A rounded label. xy is the top-left corner, or the bottom-right one
    when anchor is 'rb'."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 2 * pad[0], b - t + 2 * pad[1]
    x, y = xy
    if anchor == "rb":
        x, y = x - w, y - h
    draw.rounded_rectangle((x, y, x + w, y + h), radius=6, fill=bg)
    draw.text((x + pad[0] - l, y + pad[1] - t), text, font=font, fill=fg)


def _tile(src: Path, tag: str, badge: str, gold: bool, filled: bool) -> Image.Image:
    tile = Image.new("RGB", (TILE_W, TILE_H), (255, 255, 255))
    if src is not None and src.exists():
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((TILE_W, TILE_H), Image.LANCZOS)
            tile.paste(im, ((TILE_W - im.width) // 2, (TILE_H - im.height) // 2))
    else:
        tile = Image.new("RGB", (TILE_W, TILE_H), PANEL)
    draw = ImageDraw.Draw(tile)
    if gold:
        draw.rectangle((0, 0, TILE_W - 1, TILE_H - 1), outline=GOLD, width=6)
    _pill(draw, (16, 16), tag.upper(), _font(22, bold=True),
          GOLD if gold else (0, 0, 0), (26, 18, 0) if gold else (238, 241, 244))
    if badge:
        _pill(draw, (TILE_W - 16, TILE_H - 16), badge, _mono(20),
              FILLED if filled else (0, 0, 0), (238, 241, 244), anchor="rb")
    return tile


def render_strip(runs_dir: Path, rid: str, src: Path, run: Path, picks: dict,
                 note: str, models: str) -> Path:
    """The row as one PNG: input, best, 2nd, 3rd, 4th, and a caption bar."""
    out_dir = runs_dir / STRIP_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{rid}.png"
    tiles = [_tile(runs_dir / thumb(src, runs_dir, HERO_MAX) if src else None,
                   "input", "", False, False)]
    for rank in (1, 2, 3, 4):
        rec = picks.get(rank, {})
        name = rec.get("file") or ("best.png" if rank == 1 else f"best_{rank}.png")
        pick = run / "output" / name
        who = rec.get("chosen_by", "")
        badge = f"{rec.get('candidate', '')} {who}".strip() if rec else ""
        tiles.append(_tile(runs_dir / thumb(pick, runs_dir, HERO_MAX) if pick.exists() else None,
                           RANK_TAG[rank], badge, rank == 1, who == "harness"))
    width = TILE_W * 5 + DIVIDER * 4
    strip = Image.new("RGB", (width, TILE_H + CAPTION_H), LINE)
    x = 0
    for tile in tiles:
        strip.paste(tile, (x, 0))
        x += TILE_W + DIVIDER
    draw = ImageDraw.Draw(strip)
    draw.rectangle((0, TILE_H, width, TILE_H + CAPTION_H), fill=PANEL)
    mono = _mono(19)
    draw.text((16, TILE_H + 12), rid, font=mono, fill=DIM)
    l, _, r, _ = draw.textbbox((0, 0), note, font=mono)
    draw.text(((width - (r - l)) // 2, TILE_H + 12), note, font=mono, fill=DIM)
    right = src.name if src else "input unmatched"
    l, _, r, _ = draw.textbbox((0, 0), right, font=mono)
    draw.text((width - 16 - (r - l), TILE_H + 12), right, font=mono, fill=DIM)
    draw.text((16, TILE_H + 40), models, font=mono, fill=DIM)
    strip.save(out, "PNG", optimize=True)
    return out


def build(runs_dir: Path, inputs_dir: Path, run_ids):
    bank = input_bank(inputs_dir)
    cells = []
    strips = []
    for rid in run_ids:
        run = runs_dir / rid
        seg = run / "archive" / "source_clean.jpg"
        m = match_input(seg if seg.exists() else None, bank)
        src = m[0] if m else None
        picks = picks_by_rank(run)
        tiles = shot(src, runs_dir, "input", "in") + "".join(
            pick_tile(run, runs_dir, rank, picks) for rank in (1, 2, 3, 4))
        filled = sum(1 for r in picks.values() if r.get("chosen_by") == "harness")
        note = f"{filled} slot(s) filled by the harness" if filled else "all four chosen by the model"
        endpoint, agent, basis = models_of(run)
        models = (f"images: {short_model(endpoint) or 'unknown'}"
                  f"  \u00b7  agent: {agent or 'unknown'}")
        detail = f"{endpoint or 'image model unknown'} ({basis or 'nothing in the run names it'}); agent {agent or 'unknown'}"
        strip = render_strip(runs_dir, rid, src, run, picks, note, models)
        strips.append(strip)
        cells.append(f"""<figure class="pair">
  <div class="imgs">{tiles}</div>
  <figcaption class="cap"><span>{esc(rid)}</span><span>{esc(note)}</span><span>{esc(src.name if src else 'input unmatched')}</span>
  <a class="dl" href="{esc(rel_to(strip, runs_dir))}" download="{esc(rid)}_strip.png">this row as png</a>
  <span class="models" title="{esc(detail)}">{esc(models)}</span></figcaption>
</figure>""")

    bundle = bundle_strips(runs_dir, strips)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laydown - original and the four picks</title>
<style>{CSS}</style></head>
<body>
<header><div><h1>Laydown - original and the four picks</h1>
<p>{len(cells)} runs &middot; left is the off-set photo that went in, beside it the best pick, then the second, third and fourth
&middot; each pick names its candidate and who chose it &middot; the second caption line names the image model and the agent model
&middot; click any tile for full resolution</p></div>
<a class="dl" href="{esc(rel_to(bundle, runs_dir))}" download="laydown_strips.zip">Download all {len(cells)} rows as images (zip)</a></header>
<main>
{chr(10).join(cells)}
</main>
<div id="lb"><img alt=""></div>
<script>{JS}</script>
</body></html>
"""
    out = runs_dir / "showcase.html"
    out.write_text(doc)
    return out


def bundle_strips(runs_dir: Path, strips) -> Path:
    out = runs_dir / STRIP_DIR_NAME / "laydown_strips.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for s in strips:
            z.write(s, s.name)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ids = sys.argv[1:] or shipped_runs(root / "runs")
    if not ids:
        raise SystemExit("no run under runs/ has an output/best.png to show")
    print("wrote", build(root / "runs", root / "inputs", ids))
