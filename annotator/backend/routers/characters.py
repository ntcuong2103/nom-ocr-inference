from fastapi import APIRouter, Query

from ..index import index
from ..schemas import CharacterCount

router = APIRouter(prefix="/api/characters", tags=["characters"])


@router.get("", response_model=list[CharacterCount])
def list_characters(q: str | None = None, limit: int = Query(default=50, ge=1, le=500)):
    df = index.df
    counts = df.character.value_counts()
    if q:
        counts = counts[counts.index.str.contains(q, regex=False)]
    counts = counts.head(limit)
    return [CharacterCount(character=char, count=int(cnt)) for char, cnt in counts.items()]
