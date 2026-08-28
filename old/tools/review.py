#!/usr/bin/env python3
"""Step 5 - copy the picks to output/ at full resolution and build the sheet.

    python tools/review.py --run runs/<stamp> --picks <best>,<2nd>,<3rd>,<4th>

Order matters: the first pick is the winner and is named accordingly. Nothing is
resampled on the way to output/ - the picks are byte-for-byte the generations.
Only review_sheet.jpg is downscaled, and it is a contact sheet, never a
deliverable.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

import common as C
import cutout


def why_line(c: dict, grade_schema: bool) -> str:
    """One line of evidence for a candidate, in whichever grader's terms.

    grade_flats.py measures fidelity to the cleaned source - silhouette IoU,
    colour drift, texture distance - and construction; the measure.py it
    replaced measured colour drift and rigid dimensions. Printing one set of
    column names over the other grader's numbers is how a run ends up quoting
    figures nobody computed.
    """
    if grade_schema:
        con = [k for k in c.get("construction", []) if k.get("verdict") == "MISMATCH"]
        t = c.get("terms") or {}
        return (f"grade {c['score']:.1f}  silhouette {t.get('silhouette', 0):.0f}  "
                f"colour {t.get('colour', 0):.0f}  wrinkle {t.get('wrinkle', 0):.0f}  "
                f"bg {t.get('background', 0):.0f}"
                + (f"  construction altered in {len(con)} region(s)" if con else ""))
    return (f"score {c['score']:6.1f}  colour {c['colour_drift']:.1f}  "
            f"len {c['len_pct']:+.1f}%  top {c['topw_pct']:+.1f}%"
            + (f"  clipped {c['clip']}" if c.get("clip") else ""))


def sheet(paths, labels, out_path: Path, cell: int = 720) -> None:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell, cell), Image.LANCZOS)
        ims.append(im)
    pad, bar = 16, 34
    W = pad + sum(i.width + pad for i in ims)
    H = pad + max(i.height for i in ims) + bar + pad
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    x = pad
    for im, lab in zip(ims, labels):
        canvas.paste(im, (x, pad))
        d.text((x + 2, pad + im.height + 8), lab, fill=(20, 20, 20))
        x += im.width + pad
    canvas.save(out_path, quality=92)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--picks", required=True,
                    help="best-first, from the KEEP list grade_flats.py printed. "
                         "Full stems (cand_01,cand_04) or bare numbers "
                         "for the cand_ set (1,4).")
    ap.add_argument("--reference", type=Path, default=C.INPUTS / "reference_image.jpg")
    ap.add_argument("--no-cutout", dest="cutout", action="store_false",
                    help="skip the transparent-background PNGs")
    ap.add_argument("--trim", action="store_true",
                    help="crop each cutout to the garment. Off by default so "
                         "every layer stays registered to the original frame.")
    ap.add_argument("--target", type=int, default=4,
                    help="how many the run is meant to deliver. Falling short "
                         "is a legitimate outcome, but it hands the decision "
                         "back to a person, so output/ gets the contact sheets "
                         "and a NEEDS_REVIEW.md rather than just fewer files.")
    ap.add_argument("--force", action="store_true",
                    help="ship a candidate grade_flats.py rejected. Requires a "
                         "reason in the log.")
    a = ap.parse_args()

    arch, outd = a.run / "archive", a.run / "output"
    outd.mkdir(parents=True, exist_ok=True)

    # Accepts either the full stems measure.py prints ("probe_1_3,cand_01") or
    # bare numbers for the cand_ set ("1,4"). Stems are the safe form: a pool
    # mixing cand_ and probe_ makes a bare number ambiguous, and a probe taken
    # at the batch resolution is an eligible pick.
    stems = []
    for tok in (t.strip() for t in a.picks.split(",")):
        if not tok:
            continue
        stems.append(f"cand_{int(tok):02d}" if tok.isdigit()
                     else tok[:-4] if tok.endswith(".png") else tok)

    # A run once reasoned its way to one set of picks and then typed the numbers
    # out of an example, shipping two candidates the table had rejected for
    # colour drift. Cross-check against what was actually measured.
    metrics = arch / "metrics.json"
    if metrics.exists():
        m = json.loads(metrics.read_text())
        grade_schema = m.get("schema") == "grade"
        grader = "grade_flats.py" if grade_schema else "measure.py"
        bad = {c["cand"]: c for c in m["candidates"] if c["reject"]}
        hit = [(st, bad[st]) for st in stems if st in bad]
        if hit and not a.force:
            print(f"REFUSED. These picks were rejected by {grader}:")
            for st, c in hit:
                print(f"  {st}  {c['reject_why']}")
                print(f"      {why_line(c, grade_schema)}")
            keep = [c["cand"] for c in m["candidates"] if not c["reject"]]
            print(f"\nKEEP list: {', '.join(keep)}")
            print(f"Pick from those, or re-run with --force and justify it in "
                  f"the log.")
            return 1
        for st, c in hit:
            print(f"  WARNING: shipping rejected {st} "
                  f"({c['reject_why']}) because --force was passed.")

        # Skipping a higher-ranked eligible candidate may be right - the table
        # cannot see wrinkles or invented straps. But it has to be deliberate.
        # On one run the top-scoring image was silently left out and four
        # lower-scoring ones shipped, and the log never mentioned it.
        eligible = [c for c in m["candidates"] if not c["reject"]]
        worst = min((c["score"] for c in eligible if c["cand"] in stems),
                    default=None)
        skipped = [c for c in eligible
                   if c["cand"] not in stems and worst is not None
                   and c["score"] > worst]
        if skipped and not a.force:
            print("REFUSED. These rank ABOVE a candidate you picked, and were "
                  "left out:")
            for c in skipped:
                print(f"  {c['cand']:12} {why_line(c, grade_schema)}"
                      + ("  (NO-OP)" if c.get("noop") else ""))
            print("\nEither include them, or re-run with --force and say in "
                  "`## Notes` what you saw that the table did not. A wrinkle "
                  "ranking you disagree with is a legitimate reason - it is "
                  "batch-relative and reads form shading as easily as creases. "
                  "A construction MISMATCH is not.")
            return 1
    else:
        grade_schema = False
        print("  NOTE: no metrics.json - picks not cross-checked. "
              "Run grade_flats.py first.")

    # Short of the target? Fill it from what has already been generated, best
    # first. Generating more is the wrong answer: the images are paid for, and a
    # batch that failed the same way six times will keep failing. Rejected
    # candidates are ranked, not binary - the next-best is named here with the
    # defect it carries, so filling a place is a decision made with eyes open.
    if len(stems) < a.target and not a.force and metrics.exists():
        rest = [c for c in m["candidates"] if c["cand"] not in stems]
        if rest:
            need = a.target - len(stems)
            print(f"Only {len(stems)} of {a.target} picked. "
                  f"{len(rest)} more are already generated and paid for.")
            print(f"\nNext best, in order - each with what it would cost you:")
            for c in rest[:max(need + 2, 4)]:
                why = c["reject_why"] or ("no-op, lay unchanged"
                                          if c.get("noop") else "eligible")
                print(f"  {c['cand']:10} score {c['score']:6.1f}  {why}")
            print(f"\nLook at the contact sheets, then take the {need} you would "
                  f"defend:")
            print(f"  review.py --run {a.run} --picks "
                  f"{','.join(stems)},{','.join(c['cand'] for c in rest[:need])} "
                  f"--force")
            print(f"Record in `## Notes` what each added pick carries. Do not "
                  f"generate more - if the batch failed the same way repeatedly, "
                  f"the prompt is wrong and more draws buy more of the same.")
            return 1

    # Clear previous picks. Without this a second run leaves the first set
    # behind under different names, and output/ no longer says what shipped.
    for old in sorted(outd.glob("pick*.png")):  # includes _cutout.png
        old.unlink()

    picked = []
    for rank, st in enumerate(stems, 1):
        src = arch / f"{st}.png"
        if not src.exists():
            return print(f"Not found: {src}") or 1
        name = (f"pick{rank}_best_{st}.png" if rank == 1
                else f"pick{rank}_{st}.png")
        dst = outd / name
        shutil.copy2(src, dst)          # untouched, full resolution
        im = Image.open(dst)
        print(f"{name:32} {im.width}x{im.height}  {dst.stat().st_size/1e6:.1f} MB")
        picked.append(dst)

        # The cutout is the actual deliverable: the garment on its own layer, so
        # the retouch team sets placement, canvas and plate themselves. That is
        # why nothing in this pipeline grades framing.
        if a.cutout:
            co = outd / f"{dst.stem}_cutout.png"
            info = cutout.cut(dst, co, feather=0.0, trim=a.trim, pad=24)
            print(f"{'  -> ' + co.name:32} {info['size'][0]}x{info['size'][1]}  "
                  f"{info['mb']:.1f} MB  transparent background")

    # Short of the target, the run stops being a delivery and becomes a referral.
    # Rather than reporting "3 shipped" and leaving the rest in archive/, put
    # everything a human needs to finish the shortlist into output/: the contact
    # sheets, and every candidate's numbers and reject reason.
    if len(picked) < a.target and metrics.exists():
        m = json.loads(metrics.read_text())
        sheets = sorted(arch.glob("sheet_*.jpg")) + sorted(arch.glob("grade_results.jpg"))
        for s in sheets:
            shutil.copy2(s, outd / s.name)
        L = [f"# Needs review - {len(picked)} of {a.target} picked", "",
             f"The pipeline could not fill {a.target} places. Everything needed "
             f"to finish by hand is in this folder.", "",
             "## Contact sheets", ""]
        L += [f"- `{s.name}` - cleaned SOURCE beside each candidate, same scale"
              for s in sheets] or ["- (none built)"]
        L += ["", "## Every candidate", ""]
        if grade_schema:
            L += ["| candidate | grade | silhouette | colour | wrinkle | bg | "
                  "construction | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
            for c in m["candidates"]:
                con = [k for k in c.get("construction", [])
                       if k.get("verdict") == "MISMATCH"]
                t = c.get("terms") or {}
                # The reject reason is a paragraph per altered region. In a table
                # cell it pushes every other column off the screen, so name the
                # regions here and leave the detail to grade_results.json.
                v = ("REJECT " + ", ".join(k["region"] for k in con) if con
                     else "BELOW pass mark" if c["reject"]
                     else "PICKED" if c["cand"] in stems else "eligible")
                L.append(f"| `{c['cand']}` | {c['score']:.1f} | "
                         f"{t.get('silhouette', 0):.0f} | {t.get('colour', 0):.0f} | "
                         f"{t.get('wrinkle', 0):.0f} | "
                         f"{t.get('background', 0):.0f} | "
                         f"{'intact' if not con else str(len(con)) + ' altered'}"
                         f" | {v} |")
            L += ["", "Per-region detail for every MISMATCH is in "
                      "`archive/grade_results.json`."]
        else:
            L += ["| candidate | colour | len% | topw% | sym | seam/src | verdict |",
                  "|---|---|---|---|---|---|---|"]
            for c in m["candidates"]:
                v = ("REJECT " + c["reject_why"] if c["reject"]
                     else "PICKED" if c["cand"] in stems
                     else "NO-OP, lay unchanged" if c.get("noop") else "eligible")
                L.append(f"| `{c['cand']}` | {c['colour_drift']:.1f} | "
                         f"{c['len_pct']:+.1f} | {c['topw_pct']:+.1f} | "
                         f"{c['symmetry']:.3f} | {c['seam_vs_src']:.2f} | {v} |")
        L += ["", "## To add one by hand", "",
              "Look at the sheets, then re-run with it included:", "",
              f"    review.py --run {a.run} --picks "
              f"{','.join(stems)},<candidate>", "",
              "Add `--force` if it was rejected, and say why in `## Notes`.", ""]
        (outd / "NEEDS_REVIEW.md").write_text("\n".join(L))
        print(f"\nONLY {len(picked)} OF {a.target} PICKED.")
        print(f"  wrote {outd.name}/NEEDS_REVIEW.md and copied "
              f"{len(sheets)} contact sheet(s) here, so the shortlist can be "
              f"finished by eye.")

    # The reference is a nicety on a contact sheet, and inputs/ does not always
    # carry one - this workspace has reference_greyscale.jpg and no
    # reference_image.jpg. Opening a missing file here threw *after* the picks
    # and cutouts were already written, so the run reported a traceback on work
    # that had in fact completed.
    sp = outd / "review_sheet.jpg"
    heads, labels = [], []
    if a.reference and a.reference.exists():
        heads, labels = [a.reference], ["reference"]
    else:
        print(f"  (no reference at {a.reference} - sheet shows the picks alone)")
    sheet(heads + picked,
          labels + [f"pick{i+1} {st}" for i, st in enumerate(stems)],
          sp)
    print(f"\n{sp.name}  {Image.open(sp).size[0]}x{Image.open(sp).size[1]}")
    C.log(a.run, f"picked {len(picked)}: {a.picks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
