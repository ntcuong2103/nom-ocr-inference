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
from lcs_util import edit_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ids_dict() -> dict:
    """Load character -> IDS decomposition string mapping used to encode ground-truth text."""
    ids_dict = {}
    with open(Config.IDS_EXP, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                ids_dict[parts[0]] = parts[1]
    return ids_dict


def align_line_predictions(
    predicted_ids_list: list,
    predicted_text_list: list,
    gt_ids_list: list,
    ground_truth_text: str,
) -> list:
    """Align predicted IDS strings (one per bbox, in reading order) against the ground-truth
    line, itself encoded as one IDS string per ground-truth character.

    The pairwise cost between a bbox and a ground-truth character is the edit distance between
    their IDS strings, not a binary substring test - so a bbox whose predicted decomposition is
    only partially wrong still aligns to the right ground-truth position instead of dropping out
    of the alignment. Backtracking through this DP is what recovers the actual bbox <-> ground
    truth correspondence, which is what lets incorrect boxes still get the correct label attached
    (with selection=0) instead of falling back to their own noisy prediction.

    Args:
        predicted_ids_list: predicted IDS string per bbox - the alignment signal
        predicted_text_list: predicted composed-character text per bbox - fallback label only,
            used when a bbox can't be aligned to any ground-truth character
        gt_ids_list: ground-truth IDS string per ground-truth character
        ground_truth_text: ground-truth line text, same length/order as gt_ids_list

    Returns:
        List of length len(predicted_ids_list) of (is_exact_match, label_char):
        - is_exact_match: True iff the aligned ground-truth character's IDS is identical to the
          prediction's IDS (edit distance 0)
        - label_char: the aligned ground-truth character, or - if this bbox aligned to none -
          the bbox's own predicted text (or "" if it has none)
    """
    m = len(predicted_ids_list)
    n = len(gt_ids_list)

    cost = [[edit_distance(predicted_ids_list[i], gt_ids_list[j]) for j in range(n)] for i in range(m)]

    # dp[i][j]: min total edit distance aligning predictions[:i] with gt[:j]. Leaving a
    # prediction or a ground-truth character unaligned costs the length of its own IDS string,
    # so gaps are commensurate with substitutions instead of being free or infinitely cheap.
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + len(predicted_ids_list[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + len(gt_ids_list[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost[i - 1][j - 1],          # align prediction i with gt char j
                dp[i - 1][j] + len(predicted_ids_list[i - 1]),  # prediction i unaligned (extra box)
                dp[i][j - 1] + len(gt_ids_list[j - 1]),         # gt char j unaligned (missed box)
            )

    # Backtrack to recover which prediction aligns to which ground-truth character.
    aligned_gt_index = [None] * m
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i - 1][j - 1] + cost[i - 1][j - 1]:
            aligned_gt_index[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + len(predicted_ids_list[i - 1]):
            i -= 1
        else:
            j -= 1
    # any predictions left with i > 0 here were never aligned (j hit 0 first) - fine, they
    # default to None below.

    result = []
    for idx in range(m):
        gt_idx = aligned_gt_index[idx]
        if gt_idx is not None:
            match_score = max(len(predicted_ids_list[idx]), len(gt_ids_list[gt_idx])) - cost[idx][gt_idx]
            match_score = match_score / max(len(predicted_ids_list[idx]), len(gt_ids_list[gt_idx])) if max(len(predicted_ids_list[idx]), len(gt_ids_list[gt_idx])) > 0 else 0
            result.append((match_score, ground_truth_text[gt_idx]))
        else:
            pred_text = predicted_text_list[idx]
            result.append((0, pred_text[0] if len(pred_text) > 0 else ""))

    return result


def process_page(page_id: str, page_df_dict: dict, ids_dict: dict):
    """Process a single page and match predictions to ground truth.

    Args:
        page_id: Page identifier
        page_df_dict: Dictionary mapping bbox_id to prediction data for this page
        ids_dict: Character -> IDS decomposition string mapping, used to encode ground-truth text

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

        predictions_ids_list = []
        predictions_text_list = []
        for bbox_id in bboxes_ids:
            box_pred = page_df_dict.get(bbox_id, {})
            predictions_ids_list.append(box_pred.get("predicted_ids", ""))
            predictions_text_list.append(box_pred.get("predicted_text", ""))

        gt_ids_list = [ids_dict.get(c, c) for c in line_label["label"]]

        alignment = align_line_predictions(
            predictions_ids_list, predictions_text_list, gt_ids_list, line_label["label"]
        )
        for bid, (is_matched, matched_char) in zip(bboxes_ids, alignment):
            bbox_to_selection[bid] = is_matched
            bbox_to_selected_char[bid] = matched_char

    bboxes_gt = []
    bboxes_fp = bboxes.astype(float) / np.array([width, height, width, height])

    for bbox_id in range(len(bboxes)):
        if bbox_id in page_df_dict:
            line_idx = bbox_to_line.get(bbox_id, None)
            if line_idx is not None:
                selection = bbox_to_selection.get(bbox_id, 0)
                selected_char = bbox_to_selected_char.get(bbox_id, "")
                bboxes_gt.append((selected_char, bboxes_fp[bbox_id], selection, bbox_id))

    return bboxes_gt

_worker_ids_dict = None


def _init_worker(ids_dict: dict):
    """Pool initializer: stash the IDS dict once per worker process instead of pickling
    and sending it on every submit() call (it has ~100k entries)."""
    global _worker_ids_dict
    _worker_ids_dict = ids_dict


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
        bboxes_gt = process_page(page_id, page_df_dict, _worker_ids_dict)
        # write to file
        with open(page_out_path, "w", encoding="utf-8") as f:
            for pred, bbox, selection, bbox_id in bboxes_gt:
                bbox_str = " ".join([f"{coord:.6f}" for coord in bbox])
                f.write(f"{pred} {bbox_str} {selection} {bbox_id}\n")
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
    Config.OUTPUT_ROOT = Path("datasets/nomnaocr/pseudo_labels_v0.2")
    Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    page_ids = df["page_id"].unique()
    logger.info(f"Processing {len(page_ids)} pages")
    
    # Pre-process: group by page_id and convert to dict for fast lookup
    page_data = {}
    for page_id in tqdm(page_ids, desc="Preprocessing pages"):
        page_df = df[df["page_id"] == page_id]
        # Create a dict with bbox_id as key for O(1) lookup
        page_data[page_id] = {
            row["bbox_id"]: {"predicted_text": row["predicted_text"], "predicted_ids": row["predicted_ids"]}
            for _, row in page_df.iterrows()
        }

    ids_dict = load_ids_dict()
    logger.info(f"Loaded IDS dictionary ({len(ids_dict)} entries)")

    with ProcessPoolExecutor(max_workers=32, initializer=_init_worker, initargs=(ids_dict,)) as executor:
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
