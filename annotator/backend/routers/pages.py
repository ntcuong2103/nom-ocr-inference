from urllib.parse import quote

import imagesize
from fastapi import APIRouter, HTTPException

from ..crops import page_image_path
from ..index import index
from ..schemas import BoxOut, BoxPatchIn, PageOut
from ..store import BoxNotFoundError, PageNotFoundError, apply_edit

router = APIRouter(prefix="/api/pages", tags=["pages"])


@router.get("/{volume}/{page}", response_model=PageOut)
def get_page(volume: str, page: str):
    img_path = page_image_path(volume, page)
    if not img_path.exists():
        raise HTTPException(404, f"Page not found: {volume}/{page}")

    width, height = imagesize.get(str(img_path))
    rows = index.page_rows(volume, page).sort_values("box_id").to_dict("records")
    boxes = [
        BoxOut(
            box_id=r["box_id"],
            character=r["character"],
            x=r["x"],
            y=r["y"],
            w=r["w"],
            h=r["h"],
            selection_flag=r["selection_flag"],
        )
        for r in rows
    ]
    image_url = f"/static/images/{quote(volume)}/{quote(page)}.jpg"
    return PageOut(
        volume=volume,
        page=page,
        image_url=image_url,
        image_width=width,
        image_height=height,
        edited=index.is_edited(volume, page),
        boxes=boxes,
    )


@router.patch("/{volume}/{page}/boxes/{box_id}", response_model=BoxOut)
def patch_box(volume: str, page: str, box_id: int, body: BoxPatchIn):
    if body.character is None and body.selection_flag is None:
        raise HTTPException(400, "At least one of character/selection_flag is required")

    character = body.character
    if character is not None:
        character = character.strip()
        if any(c.isspace() for c in character):
            raise HTTPException(400, "character must not contain whitespace")

    if body.selection_flag is not None and not (0.0 <= body.selection_flag <= 1.0):
        raise HTTPException(400, "selection_flag must be between 0 and 1")

    try:
        row = apply_edit(
            volume, page, box_id, character=character, selection_flag=body.selection_flag
        )
    except PageNotFoundError:
        raise HTTPException(404, f"Page not found: {volume}/{page}")
    except BoxNotFoundError:
        raise HTTPException(404, f"Box not found: {volume}/{page}/{box_id}")

    return BoxOut(
        box_id=row["box_id"],
        character=row["character"],
        x=row["x"],
        y=row["y"],
        w=row["w"],
        h=row["h"],
        selection_flag=row["selection_flag"],
    )
