"""Path configuration for the annotation tool."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "nomnaocr"

IMAGE_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "pseudo_labels_v0.1"
LABELS_EDITED_ROOT = DATASET_ROOT / "pseudo_labels_v0.1_edited"

# Matches Config.BBOX_EXPAND_RATIO in the existing OCR pipeline (config.py)
BBOX_EXPAND_RATIO = 1.2

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
