"""Pydantic response/request models for the annotation API."""

from pydantic import BaseModel, Field


class BoxOut(BaseModel):
    box_id: int
    character: str
    x: float
    y: float
    w: float
    h: float
    selection_flag: float


class BoxPatchIn(BaseModel):
    character: str | None = None
    selection_flag: float | None = None


class PageOut(BaseModel):
    volume: str
    page: str
    image_url: str
    image_width: int
    image_height: int
    edited: bool
    boxes: list[BoxOut]


class VolumeSummary(BaseModel):
    volume: str
    num_pages: int
    num_boxes: int
    num_confirmed: int
    num_unconfirmed: int


class PageSummary(BaseModel):
    page: str
    num_boxes: int
    num_confirmed: int
    num_unconfirmed: int
    edited: bool


class CropItem(BaseModel):
    volume: str
    page: str
    box_id: int
    character: str
    selection_flag: float
    crop_url: str


class CropsPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CropItem]


class CharacterCount(BaseModel):
    character: str
    count: int
