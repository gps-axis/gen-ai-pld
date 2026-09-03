"""Convert HEIC images to JPG.

Usage:
    python scripts/heic_to_jpg.py inputs/IMG_4249.HEIC
    python scripts/heic_to_jpg.py inputs/IMG_4249.HEIC out.jpg -q 100
    python scripts/heic_to_jpg.py inputs/            # batch a folder

You do not need this before a run any more: ./run.sh --source photo.HEIC
decodes the file itself, into the run folder. This stays for making a JPEG
you want to keep - a library image, or a source to hand to someone else.

The conversion itself lives in tools/common.py (heif_to_jpeg), so the harness
and this script cannot produce two different JPEGs from the same photo.

Requires: pip install pillow-heif
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from common import heif_to_jpeg  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Convert HEIC images to JPG.")
    parser.add_argument("src", type=Path, help="HEIC file, or a folder to batch")
    parser.add_argument("dst", type=Path, nargs="?", help="output path (single file only)")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality, default 95")
    args = parser.parse_args()

    if args.src.is_dir():
        files = sorted(
            f for f in args.src.iterdir() if f.suffix.lower() in {".heic", ".heif"}
        )
        if not files:
            print(f"no HEIC files in {args.src}")
            return
        for f in files:
            out = heif_to_jpeg(f, f.with_suffix(".jpg"), args.quality)
            print(f"{f.name} -> {out.name}")
    else:
        out = args.dst or args.src.with_suffix(".jpg")
        heif_to_jpeg(args.src, out, args.quality)
        print(f"{args.src.name} -> {Path(out).name}")


if __name__ == "__main__":
    main()
