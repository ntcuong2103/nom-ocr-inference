"""Shared utility functions for OCR processing and evaluation."""

from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def process_ocr_results(ocr_csv_path: str) -> pd.DataFrame:
    """Parse OCR results from CSV file.
    
    Args:
        ocr_csv_path: Path to the OCR results CSV file
        
    Returns:
        Processed DataFrame with page_id and bbox_id columns
    """
    df = pd.read_csv(ocr_csv_path, encoding="utf-8")
    df[["page_id", "bbox_id"]] = df["image_id"].str.split("@", expand=True)
    df["page_id"] = df["page_id"].apply(
        lambda x: str(Path(Path(x).parent.name, Path(x).stem))
    )
    df["bbox_id"] = df["bbox_id"].astype(int)
    df["predicted_text"] = df["predicted_text"].fillna("")
    return df


def parse_line_labels(label_file: Path) -> List[Dict]:
    """Parse line-level ground truth labels.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        List of dictionaries containing bbox and label information
    """
    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 9:
                continue
                
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                label = parts[8]
                xs = [x1, x2, x3, x4]
                ys = [y1, y2, y3, y4]
                labels.append({
                    "bbox": (min(xs), max(xs), min(ys), max(ys)),
                    "label": label
                })
            except ValueError:
                continue
    return labels


def is_inside(bbox_center: Tuple[float, float], line_bbox: Tuple[float, float, float, float]) -> bool:
    """Check if a bbox center is inside a line bbox.
    
    Args:
        bbox_center: (x, y) coordinates of bbox center
        line_bbox: (xmin, xmax, ymin, ymax) of line bbox
        
    Returns:
        True if bbox center is inside line bbox
    """
    x_center, y_center = bbox_center
    lxmin, lxmax, lymin, lymax = line_bbox
    return (lxmin <= x_center <= lxmax) and (lymin <= y_center <= lymax)


def load_yolo_bboxes(detection_file: Path, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load and convert YOLO format bounding boxes.
    
    Args:
        detection_file: Path to YOLO format label file
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (classes, bboxes) as numpy arrays
    """
    with open(detection_file, "r") as f:
        labels = f.readlines()
    
    classes = [label.strip().split()[0] for label in labels]
    bboxes = [list(map(float, label.strip().split()[1:])) for label in labels]
    
    classes = np.array(classes)
    bboxes = np.array(bboxes)
    bboxes = bboxes * np.array([width, height, width, height])
    bboxes = bboxes.astype(int)
    
    return classes, bboxes
