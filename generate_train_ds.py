"""Generate labels for re-training character IDS recognizer"""

from ast import List
import logging
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import imagesize
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config
from utils import process_ocr_results, parse_line_labels, is_inside, load_yolo_bboxes
from lcs_util import lcs_string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_page(page_id: str, df: pd.DataFrame):
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

    lcs_strings = []
    for line_idx, line_label in enumerate(line_labels):
        bboxes_ids = line_to_bboxes[line_idx]
        bboxes_ids = sorted(bboxes_ids, key=lambda bid: bbox_center[bid][1])

        texts = []
        for bbox_id in bboxes_ids:
            row = df[(df["page_id"] == page_id) & (df["bbox_id"] == bbox_id)]
            if not row.empty:
                texts.append(row.iloc[0]["predicted_text"])

        predicted_text = "".join(texts)
        ground_truth_text = line_label["label"]
        lcs_strings.append(lcs_string(predicted_text, ground_truth_text))


    bboxes_gt = []
    bboxes_fp = bboxes.astype(float) / np.array([width, height, width, height])


    for bbox_id in range(len(bboxes)):
        row = df[(df["page_id"] == page_id) & (df["bbox_id"] == bbox_id)]
        if not row.empty:
            pred = row.iloc[0]["predicted_text"]
            line_idx = bbox_to_line.get(bbox_id, None)
            if line_idx is not None:
                selection = 0
                selected_char = pred[0] if len(pred) > 0 else ""
                for char in pred:
                    if char in lcs_strings[line_idx]:
                        selection = 1
                        selected_char = char
                        break
                
                bboxes_gt.append((selected_char, bboxes_fp[bbox_id], selection))

    return bboxes_gt

def process_single_page(page_id, df):
    """Wrapper for processing a single page."""
    page_out_path = Config.OUTPUT_ROOT / f"{page_id}.txt"
    if page_out_path.exists():
        logger.info(f"Skipping {page_id} (already processed)")
        return

    # create output directory 
    page_out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing page: {page_id}")
    try:
        bboxes_gt = process_page(page_id, df)
        # write to file
        with open(page_out_path, "w", encoding="utf-8") as f:
            for pred, bbox, selection in bboxes_gt:
                bbox_str = " ".join([f"{coord:.6f}" for coord in bbox])
                f.write(f"{pred} {bbox_str} {selection}\n")
    except Exception as e:
        logger.error(f"Error processing {page_id}: {e}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO", help="Set the logging level (e.g., INFO, WARNING, ERROR)")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format='%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(getattr(logging, args.log_level.upper()))
    
    df = process_ocr_results(str(Config.OCR_RESULTS_CSV))
    Config.OUTPUT_ROOT = Path("nomnaocr_labels_v1")
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    page_ids = df["page_id"].unique()
    logger.info(f"Processing {len(page_ids)} pages")
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_page, page_id, df): page_id for page_id in page_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pages"):
            page_id = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in multiprocessing for page {page_id}: {e}")
    
    logger.info("Generation completed")


if __name__ == "__main__":
    main()
