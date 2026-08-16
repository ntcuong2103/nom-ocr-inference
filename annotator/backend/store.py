"""Copy-on-write persistence for box edits.

An edited-output label file is always a complete, self-contained
replacement for the page (never a diff), which is what makes "prefer the
edited file, else the original" a correct merge strategy on reload.
"""

import os
import threading

from . import settings
from .index import index
from .labels import Box, format_label_line

_write_lock = threading.Lock()


class PageNotFoundError(Exception):
    pass


class BoxNotFoundError(Exception):
    pass


def apply_edit(
    volume: str,
    page: str,
    box_id: int,
    character: str | None = None,
    selection_flag: float | None = None,
) -> dict:
    with _write_lock:
        page_df = index.page_rows(volume, page)
        if page_df.empty:
            raise PageNotFoundError(f"{volume}/{page}")

        rows = page_df.sort_values("line_index").to_dict("records")
        target = None
        for row in rows:
            if row["box_id"] == box_id:
                target = row
                break
        if target is None:
            raise BoxNotFoundError(f"{volume}/{page}/{box_id}")

        if character is not None:
            target["character"] = character
        if selection_flag is not None:
            target["selection_flag"] = selection_flag

        out_dir = settings.LABELS_EDITED_ROOT / volume
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / f"{page}.txt"
        tmp_path = out_dir / f"{page}.txt.tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                box = Box(
                    box_id=row["box_id"],
                    line_index=row["line_index"],
                    character=row["character"],
                    x=row["x"],
                    y=row["y"],
                    w=row["w"],
                    h=row["h"],
                    selection_flag=row["selection_flag"],
                )
                f.write(format_label_line(box))
        os.replace(tmp_path, final_path)

        # Only commit to the in-memory index after the write has succeeded,
        # so the index never diverges from what's durably on disk.
        index.replace_page_rows(volume, page, rows)

        return target
