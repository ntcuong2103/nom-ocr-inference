"""Evaluate OCR results against ground truth."""

import logging
from collections import defaultdict
from pathlib import Path

import imagesize
import pandas as pd
import numpy as np

from config import Config
from utils import process_ocr_results, parse_line_labels, is_inside, load_yolo_bboxes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_page(page_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Process a single page and match predictions to ground truth.
    
    Args:
        page_id: Page identifier
        df: DataFrame containing OCR predictions
        
    Returns:
        DataFrame with predicted and ground truth texts for the page
    """
    detection_file = Config.DETECTION_LABELS / f"{page_id}.txt"
    image_file = Config.IMAGE_ROOT / f"{page_id}.jpg"
    line_label_file = Config.LINE_LABELS_ROOT / f"{page_id}.txt"

    width, height = imagesize.get(str(image_file))
    _, bboxes = load_yolo_bboxes(detection_file, width, height)
    line_labels = parse_line_labels(line_label_file)

    # Map bboxes to lines
    line_to_bboxes = defaultdict(list)
    bbox_to_line = {}
    
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bbox_center = np.column_stack((x + w / 2, y + h / 2))
    
    for line_idx, line_label in enumerate(line_labels):
        for bbox_id in range(len(bboxes)):
            if bbox_id in bbox_to_line:
                continue
            if is_inside(bbox_center[bbox_id], line_label["bbox"]):
                line_to_bboxes[line_idx].append(bbox_id)
                bbox_to_line[bbox_id] = line_idx

    # Process each line
    predicted_texts = []
    ground_truth_texts = []
    
    for line_idx, line_label in enumerate(line_labels):
        bboxes_ids = line_to_bboxes[line_idx]
        bboxes_ids = sorted(bboxes_ids, key=lambda bid: bbox_center[bid][1])

        texts = []
        for bbox_id in bboxes_ids:
            row = df[(df["page_id"] == page_id) & (df["bbox_id"] == bbox_id)]
            if not row.empty:
                texts.append(row.iloc[0]["predicted_text"])

        predicted_texts.append("".join(texts))
        ground_truth_texts.append(line_label["label"])
    
    return pd.DataFrame({
        "page_id": page_id,
        "predicted_text": predicted_texts,
        "ground_truth_text": ground_truth_texts,
    })


def main():
    """Main evaluation function."""
    df = process_ocr_results(str(Config.OCR_RESULTS_CSV))
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    page_ids = df["page_id"].unique()
    logger.info(f"Processing {len(page_ids)} pages")
    
    for page_id in page_ids:
        page_out_path = Config.OUTPUT_ROOT / f"{page_id}.csv"
        if page_out_path.exists():
            logger.info(f"Skipping {page_id} (already processed)")
            continue

        logger.info(f"Processing page: {page_id}")
        try:
            page_df = process_page(page_id, df)
            page_out_path.parent.mkdir(parents=True, exist_ok=True)
            page_df.to_csv(page_out_path, index=False, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error processing {page_id}: {e}")
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
