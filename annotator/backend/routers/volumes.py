from fastapi import APIRouter, HTTPException

from .. import index as index_mod
from ..index import index
from ..schemas import PageSummary, VolumeSummary

router = APIRouter(prefix="/api/volumes", tags=["volumes"])


@router.get("", response_model=list[VolumeSummary])
def list_volumes():
    df = index.df
    counts = (
        df.groupby("volume")
        .agg(
            num_pages=("page", "nunique"),
            num_boxes=("box_id", "size"),
            num_confirmed=("selection_flag", lambda s: int((s == 1).sum())),
            num_unconfirmed=("selection_flag", lambda s: int((s != 1).sum())),
        )
        .to_dict("index")
    )
    result = []
    for volume in index_mod.list_volumes():
        stats = counts.get(
            volume,
            {"num_pages": 0, "num_boxes": 0, "num_confirmed": 0, "num_unconfirmed": 0},
        )
        result.append(VolumeSummary(volume=volume, **stats))
    return result


@router.get("/{volume}/pages", response_model=list[PageSummary])
def list_pages(volume: str):
    if volume not in index_mod.list_volumes():
        raise HTTPException(404, f"Volume not found: {volume}")

    df = index.df
    vol_df = df[df.volume == volume]
    counts = (
        vol_df.groupby("page")
        .agg(
            num_boxes=("box_id", "size"),
            num_confirmed=("selection_flag", lambda s: int((s == 1).sum())),
            num_unconfirmed=("selection_flag", lambda s: int((s != 1).sum())),
        )
        .to_dict("index")
    )
    result = []
    for page in index_mod.list_pages(volume):
        stats = counts.get(page, {"num_boxes": 0, "num_confirmed": 0, "num_unconfirmed": 0})
        result.append(
            PageSummary(page=page, edited=index.is_edited(volume, page), **stats)
        )
    return result
