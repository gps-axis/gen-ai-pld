"""Convert HEIC images to JPG.

Usage:
    python scripts/heic_to_jpg.py inputs/IMG_4249.HEIC
    python scripts/heic_to_jpg.py inputs/IMG_4249.HEIC out.jpg -q 100
    python scripts/heic_to_jpg.py inputs/            # batch a folder

Requires: pip install pillow-heif
"""

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

# Adds HEIC/HEIF support to Pillow
pillow_heif.register_heif_opener()


def heic_to_jpg(input_path, output_path, quality=95):
    im = Image.open(input_path)          # open the HEIC
    im = ImageOps.exif_transpose(im)     # bake EXIF rotation into pixels
    icc = im.info.get("icc_profile")     # iPhone shots are Display P3, keep the profile
    im = im.convert("RGB")               # JPEG needs RGB (no alpha, etc.)
    im.save(output_path, "JPEG", quality=quality, icc_profile=icc, subsampling=0)
    return output_path


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
            out = heic_to_jpg(f, f.with_suffix(".jpg"), args.quality)
            print(f"{f.name} -> {out.name}")
    else:
        out = args.dst or args.src.with_suffix(".jpg")
        heic_to_jpg(args.src, out, args.quality)
        print(f"{args.src.name} -> {Path(out).name}")


if __name__ == "__main__":
    main()
