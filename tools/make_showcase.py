#!/usr/bin/env python3
"""Input-and-best grid for a chosen set of runs -> runs/showcase.html.

Reuses the contact sheet's thumbnailer and its input recovery (runs do not
archive the raw photo, so it is matched back from inputs/).

    python3 tools/make_showcase.py [run_id ...]
"""

import sys
from pathlib import Path

from make_contact_sheet import HERO_MAX, esc, input_bank, match_input, rel_to, thumb

RUNS = [
    "20260828_132259",
    "20260828_135004",
    "20260828_143935",
    "20260828_150415",
    "20260828_151331",
    "20260828_154022",
    "20260828_155806",
]

CSS = """
:root{--bg:#0e0f12;--panel:#16191e;--line:#262b33;--tx:#e8eaed;--dim:#8b939e;--gold:#f0b429}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
h1{margin:0;font-size:18px;font-weight:600}
header{padding:20px 26px 16px;border-bottom:1px solid var(--line)}
header p{margin:4px 0 0;color:var(--dim);font-size:12.5px}
main{padding:22px 26px 60px;display:grid;gap:20px;
grid-template-columns:repeat(auto-fill,minmax(420px,1fr));max-width:1700px;margin:0 auto}
.pair{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.imgs{display:grid;grid-template-columns:1fr 1fr}
.imgs a{display:block;background:#fff;aspect-ratio:3/4;position:relative;border-right:1px solid var(--line)}
.imgs a:last-child{border-right:0}
.imgs img{width:100%;height:100%;object-fit:contain;display:block}
.tag{position:absolute;top:8px;left:8px;font-size:10px;font-weight:600;letter-spacing:.5px;
text-transform:uppercase;padding:2px 7px;border-radius:4px;background:rgba(0,0,0,.62);color:#eef1f4}
.tag.out{background:var(--gold);color:#1a1200}
.cap{display:flex;justify-content:space-between;gap:10px;padding:9px 12px;
border-top:1px solid var(--line);font-size:11.5px;color:var(--dim);
font-family:ui-monospace,Menlo,monospace}
#lb{position:fixed;inset:0;background:rgba(6,7,9,.95);display:none;z-index:50;
align-items:center;justify-content:center;padding:24px}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:94vh;object-fit:contain;background:#fff;border-radius:6px}
@media(max-width:520px){main{grid-template-columns:1fr;padding:16px}}
"""

JS = """
const lb=document.getElementById('lb'),i=lb.querySelector('img');
document.querySelectorAll('main a').forEach(a=>a.addEventListener('click',e=>{
  e.preventDefault();i.src=a.getAttribute('href');lb.classList.add('on');}));
lb.addEventListener('click',()=>{lb.classList.remove('on');i.src='';});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){lb.classList.remove('on');i.src='';}});
"""


def shot(src: Path, runs_dir: Path, tag: str, cls: str) -> str:
    if src is None or not src.exists():
        return f'<a><span class="tag {cls}">{esc(tag)}</span></a>'
    return (f'<a href="{esc(rel_to(src, runs_dir))}">'
            f'<img src="{esc(thumb(src, runs_dir, HERO_MAX))}" alt="{esc(tag)}" loading="lazy">'
            f'<span class="tag {cls}">{esc(tag)}</span></a>')


def build(runs_dir: Path, inputs_dir: Path, run_ids):
    bank = input_bank(inputs_dir)
    cells = []
    for rid in run_ids:
        run = runs_dir / rid
        seg = run / "archive" / "source_clean.jpg"
        best = run / "output" / "best.png"
        m = match_input(seg if seg.exists() else None, bank)
        src = m[0] if m else None
        cells.append(f"""<figure class="pair">
  <div class="imgs">{shot(src, runs_dir, 'input', 'in')}{shot(best if best.exists() else None, runs_dir, 'best', 'out')}</div>
  <figcaption class="cap"><span>{esc(rid)}</span><span>{esc(src.name if src else 'input unmatched')}</span></figcaption>
</figure>""")

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laydown - input and result</title>
<style>{CSS}</style></head>
<body>
<header><h1>Laydown - input and result</h1>
<p>{len(cells)} runs &middot; left is the off-set photo that went in, right is the shipped laydown &middot;
click either for full resolution</p></header>
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


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ids = sys.argv[1:] or RUNS
    print("wrote", build(root / "runs", root / "inputs", ids))
