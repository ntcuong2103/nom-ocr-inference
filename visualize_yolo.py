
#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def char_line_to_box(
    line: str, img_w: int, img_h: int
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Parse one label line in the format:
        character x y w h [conf]

    - character: UTF-8 string (e.g., Chinese character), not an ID
    - x, y: center coordinates, normalized [0..1] (common) or pixels
    - w, h: width/height, normalized [0..1] (common) or pixels

    Returns (character, (x1,y1,x2,y2)) in pixel coordinates, or None for bad lines.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    ch = parts[0]
    try:
        x, y, w, h = map(float, parts[1:5])
    except ValueError:
        return None

    # Determine if normalized (typical) or pixel-based centers
    normalized = max(abs(x), abs(y), abs(w), abs(h)) <= 1.5

    if normalized:
        xc = x * img_w
        yc = y * img_h
        bw = w * img_w
        bh = h * img_h
    else:
        xc, yc, bw, bh = x, y, w, h

    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))

    # Clamp to image bounds
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    # Filter degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return None

    return ch, (x1, y1, x2, y2)


def str_color(key: str) -> Tuple[int, int, int]:
    """Deterministic bright BGR color from a string via HSV hue hashing."""
    h = (hash(key) % 180)
    color_hsv = np.uint8([[[h, 200, 255]]])  # H,S,V for OpenCV
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])


def draw_unicode_label_pil(
    img_bgr: np.ndarray,
    text: str,
    x1: int,
    y1: int,
    color_bgr: Tuple[int, int, int],
    font: Optional["ImageFont.FreeTypeFont"],
    pad: int = 2,
) -> np.ndarray:
    """
    Draw UTF-8 text using PIL so non-ASCII (e.g., Chinese) renders properly.
    Returns the modified BGR image (numpy array).
    """

    # Convert BGR -> RGB for PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Measure text size
    tw, th = draw.textlength(text, font=font), font.getbbox(text)[3] - font.getbbox(text)[1]
    bx1, by1 = x1, max(0, y1 - th - 2 * pad)
    bx2, by2 = x1 + int(tw) + 2 * pad, y1

    fill_color = (0, 0, 255, 240) 
    draw.text((x1 + pad, by1 + pad), text, fill=fill_color, font=font)

    # Convert back RGB -> BGR
    img_bgr_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr_out


def visualize_image(
    img_path: Path,
    label_path: Optional[Path],
    thickness: int = 2,
    font_path: Optional[Path] = None,
    font_size: int = 22,
) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    h, w = img.shape[:2]

    # Prepare PIL font if available and provided
    pil_font = None
    if font_path is not None:
        try:
            pil_font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:  # pragma: no cover - font issues
            print(f"[warn] Could not load font '{font_path}': {e}", file=sys.stderr)
            pil_font = None

    if label_path and label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
        for ln in lines:
            parsed = char_line_to_box(ln, w, h)
            if parsed is None:
                continue
            ch, (x1, y1, x2, y2) = parsed
            color = str_color(ch)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            img = draw_unicode_label_pil(img, ch, x1, y1, color, pil_font)
    else:
        # No labels found; annotate image name lightly (ASCII)
        color = (255, 255, 255)
        cv2.putText(
            img,
            img_path.name,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return img


def collect_images(images_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    return imgs


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Visualize YOLO-style annotations with UTF-8 characters: "
            "each line as 'character x y w h'"
        )
    )
    ap.add_argument("--images", "-i", type=Path, required=True, help="Directory with images")
    ap.add_argument(
        "--labels", "-l", type=Path, required=True, help="Directory with .txt label files"
    )
    ap.add_argument(
        "--save-dir",
        "-o",
        type=Path,
        default=Path("viz"),
        help="Output directory to save visualizations",
    )
    ap.add_argument("--show", action="store_true", help="Open a window to preview each image")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders of --images")
    ap.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    ap.add_argument("--thickness", type=int, default=1, help="Box line thickness")
    ap.add_argument(
        "--font",
        type=Path,
        default="nom-ids/assets/NotoSansCJKjp-Regular.otf",
        help="Path to a TTF/OTF font that supports Chinese (e.g., NotoSansCJK).",
    )
    ap.add_argument("--font-size", type=int, default=12, help="Font size for labels")
    args = ap.parse_args()

    if not args.images.exists():
        ap.error(f"--images not found: {args.images}")
    if not args.labels.exists():
        ap.error(f"--labels not found: {args.labels}")
    if args.font is None:
        print(
            "[warn] Pillow not available and no font provided; CJK labels may not render.",
            file=sys.stderr,
        )

    args.save_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.images, args.recursive)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    if not images:
        print("[warn] No images found.", file=sys.stderr)
        return

    print(f"[info] Found {len(images)} images. Drawing boxes...")

    for idx, img_path in enumerate(images, 1):
        rel = img_path.relative_to(args.images)
        lbl_path = args.labels / rel.with_suffix(".txt")
        out_path = args.save_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            vis = visualize_image(
                img_path,
                lbl_path,
                thickness=args.thickness,
                font_path=args.font,
                font_size=args.font_size,
            )
        except Exception as e:
            print(f"[error] {img_path}: {e}", file=sys.stderr)
            continue

        cv2.imwrite(str(out_path), vis)

        if args.show:
            cv2.imshow("viz", vis)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # ESC
                break

        if idx % 50 == 0:
            print(f"[info] Processed {idx}/{len(images)}")

    if args.show:
        cv2.destroyAllWindows()

    print(f"[done] Saved visualizations to: {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()