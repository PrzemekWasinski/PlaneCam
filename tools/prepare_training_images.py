#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from PIL import Image
    BACKEND = "pillow"
except ImportError:
    try:
        import cv2
        BACKEND = "opencv"
    except ImportError:
        print("This script requires Pillow or OpenCV (cv2).", file=sys.stderr)
        sys.exit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def center_crop_size(width: int, height: int, aspect_w: int, aspect_h: int):
    target_ratio = aspect_w / aspect_h
    current_ratio = width / height

    if current_ratio > target_ratio:
        crop_h = height
        crop_w = int(round(height * target_ratio))
    else:
        crop_w = width
        crop_h = int(round(width / target_ratio))

    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return left, top, crop_w, crop_h


def process_with_pillow(src: Path, dst: Path, aspect_w: int, aspect_h: int, resize_w: int | None, resize_h: int | None):
    with Image.open(src) as image:
        image = image.convert("RGB")
        left, top, crop_w, crop_h = center_crop_size(image.width, image.height, aspect_w, aspect_h)
        image = image.crop((left, top, left + crop_w, top + crop_h))
        if resize_w and resize_h:
            image = image.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, quality=95)


def process_with_opencv(src: Path, dst: Path, aspect_w: int, aspect_h: int, resize_w: int | None, resize_h: int | None):
    image = cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"Unable to read image: {src}")
    height, width = image.shape[:2]
    left, top, crop_w, crop_h = center_crop_size(width, height, aspect_w, aspect_h)
    image = image[top:top + crop_h, left:left + crop_w]
    if resize_w and resize_h:
        image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), image):
        raise RuntimeError(f"Unable to write image: {dst}")


def process_image(src: Path, dst: Path, aspect_w: int, aspect_h: int, resize_w: int | None, resize_h: int | None):
    if BACKEND == "pillow":
        process_with_pillow(src, dst, aspect_w, aspect_h, resize_w, resize_h)
    else:
        process_with_opencv(src, dst, aspect_w, aspect_h, resize_w, resize_h)



def main():
    parser = argparse.ArgumentParser(description="Center-crop a folder of images to a target aspect ratio.")
    parser.add_argument("input_dir", type=Path, help="Folder containing source images")
    parser.add_argument("output_dir", type=Path, help="Folder to write processed images into")
    parser.add_argument("--aspect", default="4:3", help="Aspect ratio to crop to, default 4:3")
    parser.add_argument("--resize", default="", help="Optional output size like 640x480")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    if ":" not in args.aspect:
        raise SystemExit("--aspect must look like 4:3")
    aspect_w, aspect_h = [int(part) for part in args.aspect.split(":", 1)]

    resize_w = resize_h = None
    if args.resize:
        if "x" not in args.resize.lower():
            raise SystemExit("--resize must look like 640x480")
        resize_w, resize_h = [int(part) for part in args.resize.lower().split("x", 1)]

    count = 0
    for src in iter_images(args.input_dir):
        rel = src.relative_to(args.input_dir)
        dst = args.output_dir / rel
        process_image(src, dst, aspect_w, aspect_h, resize_w, resize_h)
        count += 1
        if count % 25 == 0:
            print(f"Processed {count} images...")

    print(f"Done. Processed {count} images from {args.input_dir} to {args.output_dir} using {BACKEND}.")


if __name__ == "__main__":
    main()
