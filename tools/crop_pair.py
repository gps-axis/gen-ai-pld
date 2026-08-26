#!/usr/bin/env python3
"""Matching crop boxes for two images that do not share a coordinate system.

    python tools/crop_pair.py --run runs/<stamp> --cand 10 --at waistband

The source photo and a candidate have different pixel dimensions AND the garment
sits in a different place in each, so the same pixel box lands on different parts
of the garment. A real run hand-picked (1700,1300) on the source and (950,700) on
a candidate believing they matched; in garment-relative terms those are 28%
across / 23% down versus 1% across / 9% down - the hip against the waistband.
Every construction verdict that run produced was comparing different things.

This locates the garment in each image with the same mask the metrics use, then
converts a garment-relative region into real pixel boxes for each. Paste the two
printed lines straight into compare_images.

The region names are per garment - `straps`, `cups`, `band` for a bra;
`waistband`, `crotch`, `hem` for legwear - and the set is chosen from the
garment_type in <run>/reference_selection.json, the same way grade_flats.py
picks its stage-3 crops. Run without --at to print the names for this run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import common as C

# Normalised to the garment's own bounding box: (u0, v0, u1, v1), origin at its
# top-left, 1.0 = its full width/height.
#
# The names are garment-aware, because a region name that means nothing on the
# garment in front of you is worse than no name at all: "waistband" on a bra is
# empty plate above the straps, and a crop of empty plate against empty plate
# compares clean and says nothing. The profile is read from the run's own
# garment_type, so the names offered always belong to the garment being graded.

# Geometric bands, true of any laydown, always available under any profile.
SHARED = {
    "top":    (0.00, 0.00, 1.00, 0.18),
    "upper":  (0.00, 0.08, 1.00, 0.38),
    "middle": (0.00, 0.36, 1.00, 0.64),
    "lower":  (0.00, 0.62, 1.00, 0.92),
    "bottom": (0.00, 0.82, 1.00, 1.00),
    "centre": (0.32, 0.36, 0.68, 0.64),
    "left":   (0.00, 0.25, 0.30, 0.75),
    "right":  (0.70, 0.25, 1.00, 0.75),
}

# Per-profile names override the shared ones, so "centre" means the centre front
# of a bra and the inseam line of a legging without the caller tracking which.
REGIONS_BY_PROFILE = {
    "leggings": {
        "waistband": (0.10, 0.00, 0.90, 0.13),
        "hip":       (0.05, 0.13, 0.95, 0.32),
        "crotch":    (0.25, 0.28, 0.75, 0.44),
        "thigh":     (0.10, 0.35, 0.90, 0.55),
        "knee":      (0.10, 0.55, 0.90, 0.72),
        "hem":       (0.05, 0.88, 0.95, 1.00),
        "centre":    (0.30, 0.42, 0.70, 0.62),
        "left":      (0.00, 0.30, 0.30, 0.70),
        "right":     (0.70, 0.30, 1.00, 0.70),
    },
    # A bra is wider than tall and hangs the other way up: the straps are at the
    # top, the band at the bottom, and there is no waistband, crotch or hem at
    # all. These match the bands grade_flats.py's stage 3 uses, so a flag it
    # raises can be looked at here under the same name.
    "bras": {
        "straps":    (0.10, 0.00, 0.90, 0.26),
        "neckline":  (0.20, 0.02, 0.80, 0.28),
        "cups":      (0.08, 0.24, 0.92, 0.64),
        "centre":    (0.32, 0.26, 0.68, 0.62),
        "band":      (0.05, 0.60, 0.95, 1.00),
        # The side panels, where the strap meets the cup and the band closes -
        # 'underarm' on a spec sheet, and one box can only be one side of it.
        "left":      (0.00, 0.18, 0.30, 0.78),
        "right":     (0.70, 0.18, 1.00, 0.78),
    },
    # Hangs the same way up as a bra - collar at the top, hem at the bottom -
    # but the sleeves push the bounding box out sideways, so the BODY occupies
    # only the middle ~55% of its width. Every body region is inset to match:
    # 'chest' taken full-width, the way the bra's 'cups' is, would be half
    # sleeve and half plate and would compare clean while the chest print was
    # wrong. Assumes the standard PDP laydown, sleeves angled down and out; if
    # a brand lays them straight out horizontally the body narrows further and
    # these need re-measuring against that library.
    "pullovers": {
        "collar":    (0.34, 0.00, 0.66, 0.13),
        "neckline":  (0.28, 0.00, 0.72, 0.19),
        "shoulders": (0.16, 0.05, 0.84, 0.27),
        "chest":     (0.24, 0.18, 0.76, 0.50),
        "body":      (0.22, 0.32, 0.78, 0.80),
        "hem":       (0.20, 0.84, 0.80, 1.00),
        "centre":    (0.34, 0.30, 0.66, 0.60),
        # The sleeves, cuff included. Named left/right rather than 'sleeve' for
        # the same reason the bra has no single 'underarm': one box is one side.
        "left":      (0.00, 0.14, 0.32, 0.80),
        "right":     (0.68, 0.14, 1.00, 0.80),
    },
    # The pullover's shape with the loft added and the hardware named. Three
    # things differ, and each one is a box the pullover set does not have:
    #
    #   * The pile stands off the seam, so the body reads ~2% of the bbox wider
    #     each side. Every body box here is the pullover's opened by that much;
    #     borrowing the tighter ones puts the fleece's own edge on the boundary,
    #     which is where a redrawn seam hides.
    #   * 'placket' is the zip line, top to hem. On a full-zip it is the single
    #     most-redrawn feature on the garment - teeth spacing, the pull, where
    #     the tape meets the collar - and no horizontal band isolates it.
    #   * 'pockets' is the hand-pocket band. The openings are diagonal welts at
    #     hip height; inside 'body' they are four percent of the frame and a
    #     judge will not mention them.
    #
    # 'collar' is taller and wider than the pullover's: a stand collar or a hood
    # occupies height a crewneck does not. On a fleece with neither, the box
    # simply shows the neckline - the same way 'pockets' costs nothing on a
    # fleece without them.
    "fleeces": {
        "collar":    (0.30, 0.00, 0.70, 0.16),
        "hood":      (0.24, 0.00, 0.76, 0.22),
        "neckline":  (0.26, 0.02, 0.74, 0.20),
        "shoulders": (0.14, 0.06, 0.86, 0.30),
        "chest":     (0.22, 0.20, 0.78, 0.52),
        "placket":   (0.40, 0.06, 0.60, 0.92),
        "pockets":   (0.20, 0.52, 0.80, 0.78),
        "body":      (0.20, 0.34, 0.80, 0.82),
        "hem":       (0.18, 0.84, 0.82, 1.00),
        "centre":    (0.32, 0.32, 0.68, 0.62),
        "left":      (0.00, 0.14, 0.32, 0.82),
        "right":     (0.68, 0.14, 1.00, 0.82),
    },
}


def bbox(path: Path, like: Path | None = None) -> tuple[tuple[int, int, int, int], str, bool]:
    """Garment bounding box in the image's own full-resolution pixels.

    `like` is the other side of the pair - the cleaned source - and turns on the
    credibility check in common.garment_box. A region is a FRACTION of this box,
    so a box that is wrong by 10x does not produce a slightly wrong crop, it
    produces a confident close-up of a different part of the garment, which is
    the exact failure this whole file exists to prevent.
    """
    g = C.garment_box(path, like=like)
    if g["box"] is None:
        raise SystemExit(f"No garment found in {path}")
    return g["box"], g["note"], g["ok"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--cand", type=int, help="candidate number, with --run")
    ap.add_argument("-a", "--image-a", type=Path, default=None,
                    help="the reference side of the pair. Defaults to "
                         "<run>/archive/offset_upload.jpg, the CLEANED image "
                         "the generator was actually given. Do not point it at "
                         "inputs/off_set_image.jpg: the raw input still carries "
                         "the hang tag, so a crop taken against it shows the "
                         "correctly-removed tag as a difference.")
    ap.add_argument("-b", "--image-b", type=Path,
                    help="second image, if not using --run/--cand")
    ap.add_argument("--at", default=None,
                    help="a region name - the set depends on the garment, run "
                         "with no --at to see it - or four normalised numbers "
                         "'u0,v0,u1,v1' inside the garment bbox")
    ap.add_argument("--profile", choices=sorted(REGIONS_BY_PROFILE), default=None,
                    help="which garment the region names describe. Read "
                         "automatically from the garment_type in "
                         "<run>/reference_selection.json; pass this to override "
                         "it, or when working without --run.")
    ap.add_argument("--max-px", type=int, default=1000,
                    help="largest crop side to allow. Above ~1024 the vision "
                         "call downscales and the look stops being 1:1, so both "
                         "boxes shrink around the region centre until they fit.")
    a = ap.parse_args()

    a_path = a.image_a
    if a_path is None:
        if not a.run:
            return print("Need --image-a, or --run so the cleaned upload can "
                         "be found.") or 1
        a_path = a.run / "archive" / "offset_upload.jpg"
    b_path = a.image_b
    if b_path is None:
        if not (a.run and a.cand):
            return print("Need --image-b, or --run with --cand.") or 1
        b_path = a.run / "archive" / f"cand_{a.cand:02d}.png"
    for p in (a_path, b_path):
        if not p.exists():
            return print(f"Not found: {p}") or 1

    # The region names follow the garment, so resolve the profile before any of
    # them mean anything. With no --run there is nothing to read it from, hence
    # the explicit fallback rather than a silent leggings default.
    if a.run:
        profile, why, prof_ok = C.garment_profile(a.run, a.profile)
    else:
        profile = a.profile or C.DEFAULT_PROFILE
        prof_ok = bool(a.profile)
        why = (f"{profile} - forced by --profile" if a.profile else
               "WARNING: no --run to read garment_type from")
    if profile not in REGIONS_BY_PROFILE:
        return print(f"unknown profile {profile!r}; choose from "
                     f"{', '.join(sorted(REGIONS_BY_PROFILE))}") or 1
    print(f"  profile: {why}" + ("" if prof_ok else
          " - named regions withheld"))

    # Named regions are withheld when the garment was never identified, rather
    # than offered on an assumption. 'waistband' on an unknown garment is a
    # guess dressed as an anatomy, and the whole point of this file is that a
    # crop taken on a guess produces verdicts about the wrong part of the
    # garment. The geometric bands describe any laydown, so those stay.
    regions = {**SHARED, **REGIONS_BY_PROFILE[profile]} if prof_ok else dict(SHARED)
    named = (f"{profile} regions" if prof_ok else
             "garment-neutral bands only, until --profile says which garment")

    if a.at is None:
        return print(f"\n--at is required. {named}: {' '.join(sorted(regions))}\n"
                     f"or four normalised numbers 'u0,v0,u1,v1'.") or 1
    if a.at in regions:
        u0, v0, u1, v1 = regions[a.at]
    else:
        try:
            u0, v0, u1, v1 = (float(x) for x in a.at.replace(" ", "").split(","))
        except ValueError:
            owners = [p for p, r in REGIONS_BY_PROFILE.items() if a.at in r]
            if owners and not prof_ok:
                return print(f"REFUSING: '{a.at}' is a {'/'.join(owners)} region "
                             f"and this run's garment was never identified "
                             f"(see the line above).\n"
                             f"  Pass --profile "
                             f"{' | '.join(sorted(REGIONS_BY_PROFILE))} to say "
                             f"which garment, or use a band that is true of any "
                             f"laydown ({' '.join(sorted(SHARED))}).") or 1
            hint = (f"  ('{a.at}' is a {'/'.join(owners)} region, and this run's "
                    f"garment is a {profile[:-1]}.)" if owners else "")
            return print(f"--at must be one of {named} "
                         f"({' '.join(sorted(regions))}) or 'u0,v0,u1,v1'.\n"
                         f"{hint}") or 1

    boxes = {}
    # Image A is the reference the check is made against, so it is measured on
    # its own; image B is checked against it.
    for label, p, like in (("a", a_path, None), ("b", b_path, a_path)):
        (x0, y0, x1, y1), note, ok = bbox(p, like=like)
        gw, gh = x1 - x0, y1 - y0
        boxes[label] = [x0 + u0 * gw, y0 + v0 * gh, (u1 - u0) * gw, (v1 - v0) * gh]
        print(f"  {p.name:22} {Image.open(p).size[0]}x{Image.open(p).size[1]}  "
              f"garment {gw}x{gh} at ({x0},{y0})")
        if not ok:
            print(f"  {'':22} {note}")

    # Both crops must stay under max_px or the vision call resamples them and the
    # comparison is no longer 1:1. Shrink both by the SAME normalised factor, so
    # they keep showing the same part of the garment.
    worst = max(max(w, h) for _, _, w, h in boxes.values())
    if worst > a.max_px:
        k = a.max_px / worst
        for v in boxes.values():
            cx, cy = v[0] + v[2] / 2, v[1] + v[3] / 2
            v[2], v[3] = v[2] * k, v[3] * k
            v[0], v[1] = cx - v[2] / 2, cy - v[3] / 2
        print(f"  shrunk both by {k:.2f} to keep each crop 1:1 under "
              f"{a.max_px}px")

    print(f"\nregion '{a.at}'  ({u0:.2f},{v0:.2f})-({u1:.2f},{v1:.2f}) "
          f"of the garment\n")
    for label in ("a", "b"):
        x, y, w, h = (int(round(v)) for v in boxes[label])
        print(f"box_{label}=\"{x},{y},{w},{h}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
