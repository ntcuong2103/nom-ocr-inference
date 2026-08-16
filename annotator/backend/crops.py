"""Character crop generation from page images.

Mirrors the square, expand-ratio crop convention used by the existing
OCR pipeline (see ImageDataset/ImageDatasetBBox in data.py, expand_ratio=1.2).
"""

from functools import lru_cache

from PIL import Image

from . import settings


@lru_cache(maxsize=64)
def _load_page_image(image_path_str: str) -> Image.Image:
    return Image.open(image_path_str).convert("RGB")


def page_image_path(volume: str, page: str):
    return settings.IMAGE_ROOT / volume / f"{page}.jpg"


def crop_box(
    volume: str,
    page: str,
    x: float,
    y: float,
    w: float,
    h: float,
    expand_ratio: float = settings.BBOX_EXPAND_RATIO,
) -> Image.Image | None:
    img_path = page_image_path(volume, page)
    if not img_path.exists():
        return None
    image = _load_page_image(str(img_path))
    iw, ih = image.width, image.height

    cx, cy = x * iw, y * ih
    bw, bh = w * iw, h * ih
    side = max(bw, bh) * expand_ratio

    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(iw, int(cx + side / 2))
    y2 = min(ih, int(cy + side / 2))

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))
