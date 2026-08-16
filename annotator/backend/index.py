"""In-memory index of all boxes across the dataset, built once at startup."""

import threading

import pandas as pd

from . import settings
from .labels import parse_label_file

COLUMNS = [
    "volume",
    "page",
    "box_id",
    "line_index",
    "character",
    "x",
    "y",
    "w",
    "h",
    "selection_flag",
]


def resolve_label_path(volume: str, page: str):
    edited = settings.LABELS_EDITED_ROOT / volume / f"{page}.txt"
    if edited.exists():
        return edited
    original = settings.LABELS_ROOT / volume / f"{page}.txt"
    if original.exists():
        return original
    return None


def list_volumes() -> list[str]:
    if not settings.IMAGE_ROOT.exists():
        return []
    return sorted(p.name for p in settings.IMAGE_ROOT.iterdir() if p.is_dir())


def list_pages(volume: str) -> list[str]:
    vol_dir = settings.IMAGE_ROOT / volume
    if not vol_dir.exists():
        return []
    return sorted(p.stem for p in vol_dir.glob("*.jpg"))


def build_index() -> pd.DataFrame:
    rows = []
    for volume in list_volumes():
        for page in list_pages(volume):
            label_path = resolve_label_path(volume, page)
            if label_path is None:
                continue
            for box in parse_label_file(label_path):
                rows.append(
                    (
                        volume,
                        page,
                        box.box_id,
                        box.line_index,
                        box.character,
                        box.x,
                        box.y,
                        box.w,
                        box.h,
                        box.selection_flag,
                    )
                )
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)


class BoxIndex:
    """Holds the live DataFrame and provides thread-safe read/replace access."""

    def __init__(self):
        self._lock = threading.Lock()
        self._df = build_index()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def is_edited(self, volume: str, page: str) -> bool:
        return (settings.LABELS_EDITED_ROOT / volume / f"{page}.txt").exists()

    def page_rows(self, volume: str, page: str) -> pd.DataFrame:
        df = self._df
        return df[(df.volume == volume) & (df.page == page)]

    def replace_page_rows(self, volume: str, page: str, rows: list[dict]) -> None:
        """Atomically swap out a page's rows for freshly-written ones."""
        with self._lock:
            df = self._df
            keep_mask = ~((df.volume == volume) & (df.page == page))
            new_rows = pd.DataFrame(rows, columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
            self._df = pd.concat([df[keep_mask], new_rows], ignore_index=True)


index = BoxIndex()
