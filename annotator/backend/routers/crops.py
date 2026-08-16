import io
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import crops as crops_mod
from ..index import index
from ..schemas import CropItem, CropsPage

router = APIRouter(prefix="/api/crops", tags=["crops"])


@router.get("", response_model=CropsPage)
def list_crops(
    selection_flag: float | None = Query(default=None, ge=0, le=1),
    character: str | None = None,
    volume: str | None = None,
    page: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    df = index.df
    mask = None

    def and_mask(cond):
        nonlocal mask
        mask = cond if mask is None else (mask & cond)

    if selection_flag is not None:
        and_mask(df.selection_flag == selection_flag)
    if character is not None:
        and_mask(df.character == character)
    if volume is not None:
        and_mask(df.volume == volume)
    if page is not None:
        and_mask(df.page == page)

    filtered = df[mask] if mask is not None else df
    filtered = filtered.sort_values(["volume", "line_index"])
    total = len(filtered)
    page_slice = filtered.iloc[offset : offset + limit]

    items = [
        CropItem(
            volume=r["volume"],
            page=r["page"],
            box_id=r["box_id"],
            character=r["character"],
            selection_flag=r["selection_flag"],
            crop_url=f"/api/crops/{quote(r['volume'])}/{quote(r['page'])}/{r['box_id']}",
        )
        for r in page_slice.to_dict("records")
    ]
    return CropsPage(total=total, limit=limit, offset=offset, items=items)


@router.get("/{volume}/{page}/{box_id}")
def get_crop(volume: str, page: str, box_id: int):
    df = index.df
    row = df[(df.volume == volume) & (df.page == page) & (df.box_id == box_id)]
    if row.empty:
        raise HTTPException(404, f"Box not found: {volume}/{page}/{box_id}")
    r = row.iloc[0]

    image = crops_mod.crop_box(volume, page, r.x, r.y, r.w, r.h)
    if image is None:
        raise HTTPException(404, f"Could not crop box: {volume}/{page}/{box_id}")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")
