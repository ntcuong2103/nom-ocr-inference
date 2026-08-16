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


def align_line_predictions(predictions_list: list, ground_truth_text: str) -> list:
    """Align a list of predicted strings (one per bbox) with the ground truth text.
    
    Args:
        predictions_list: List of predicted strings (e.g. ["明", "日"])
        ground_truth_text: Ground truth line text (e.g. "明日")
        
    Returns:
        List of tuples (is_matched, matched_char) corresponding to each prediction in predictions_list
    """
    m = len(predictions_list)
    n = len(ground_truth_text)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        pred_str = predictions_list[i - 1]
        for j in range(1, n + 1):
            gt_char = ground_truth_text[j - 1]
            if gt_char in pred_str:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    # Backtrack to find the matches
    matched = [False] * m
    matched_chars = [pred_str[0] if len(pred_str) > 0 else "" for pred_str in predictions_list]
    
    i, j = m, n
    while i > 0 and j > 0:
        pred_str = predictions_list[i - 1]
        gt_char = ground_truth_text[j - 1]
        if gt_char in pred_str:
            matched[i - 1] = True
            matched_chars[i - 1] = gt_char
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
            
    return list(zip(matched, matched_chars))


def process_page(page_id: str, page_df_dict: dict):
    """Process a single page and match predictions to ground truth.
    
    Args:
        page_id: Page identifier
        page_df_dict: Dictionary mapping bbox_id to prediction data for this page
        
    Returns:
        List of tuples with predicted characters, bboxes, and selection flags
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
    # In YOLO format, (x, y) are the actual center coordinates in pixels.
    bbox_center = np.column_stack((x, y))

    for line_idx, line_label in enumerate(line_labels):
        for bbox_id in range(len(bboxes)):
            if bbox_id in bbox_to_line:
                continue
            if is_inside(bbox_center[bbox_id], line_label["bbox"]):
                line_to_bboxes[line_idx].append(bbox_id)
                bbox_to_line[bbox_id] = line_idx

    bbox_to_selection = {}
    bbox_to_selected_char = {}

    for line_idx, line_label in enumerate(line_labels):
        bboxes_ids = line_to_bboxes[line_idx]
        bboxes_ids = sorted(bboxes_ids, key=lambda bid: bbox_center[bid][1])

        predictions_list = []
        for bbox_id in bboxes_ids:
            pred_text = page_df_dict[bbox_id]["predicted_text"] if bbox_id in page_df_dict else ""
            predictions_list.append(pred_text)

        alignment = align_line_predictions(predictions_list, line_label["label"])
        for bid, (is_matched, matched_char) in zip(bboxes_ids, alignment):
            bbox_to_selection[bid] = 1 if is_matched else 0
            bbox_to_selected_char[bid] = matched_char

    bboxes_gt = []
    bboxes_fp = bboxes.astype(float) / np.array([width, height, width, height])

    for bbox_id in range(len(bboxes)):
        if bbox_id in page_df_dict:
            line_idx = bbox_to_line.get(bbox_id, None)
            if line_idx is not None:
                selection = bbox_to_selection.get(bbox_id, 0)
                selected_char = bbox_to_selected_char.get(bbox_id, "")
                bboxes_gt.append((selected_char, bboxes_fp[bbox_id], selection))

    return bboxes_gt

def process_single_page(page_id, page_df_dict):
    """Wrapper for processing a single page."""
    page_out_path = Config.OUTPUT_ROOT / f"{page_id}.txt"
    if page_out_path.exists():
        logger.info(f"Skipping {page_id} (already processed)")
        return

    # create output directory 
    page_out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing page: {page_id}")
    try:
        bboxes_gt = process_page(page_id, page_df_dict)
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
    
    # Pre-process: group by page_id and convert to dict for fast lookup
    page_data = {}
    for page_id in tqdm(page_ids, desc="Preprocessing pages"):
        page_df = df[df["page_id"] == page_id]
        # Create a dict with bbox_id as key for O(1) lookup
        page_data[page_id] = {
            row["bbox_id"]: {"predicted_text": row["predicted_text"]}
            for _, row in page_df.iterrows()
        }
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_page, page_id, page_data[page_id]): page_id for page_id in page_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pages"):
            page_id = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in multiprocessing for page {page_id}: {e}")
    
    logger.info("Generation completed")


if __name__ == "__main__":
    main()
