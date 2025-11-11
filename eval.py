from collections import defaultdict
from pathlib import Path

import imagesize
import pandas as pd
import numpy as np

import editdistance


def process_ocr_results(ocr_csv_path):
    df = pd.read_csv(ocr_csv_path, encoding="utf-8")
    df[["page_id", "bbox_id"]] = df["image_id"].str.split("@", expand=True)
    df["page_id"] = df["page_id"].apply(
        lambda x: str(Path(Path(x).parent.name, Path(x).stem))
    )
    df["bbox_id"] = df["bbox_id"].astype(int)
    df["predicted_text"] = df["predicted_text"].fillna("")
    return df


def parse_line_labels(label_file):
    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 9:
                try:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                    label = parts[8]
                    xs = [x1, x2, x3, x4]
                    ys = [y1, y2, y3, y4]
                    xmin = min(xs)
                    xmax = max(xs)
                    ymin = min(ys)
                    ymax = max(ys)
                    labels.append({"bbox": (xmin, xmax, ymin, ymax), "label": label})
                except ValueError:
                    continue
    return labels



def is_inside(bbox_center, line_bbox):
    x_center, y_center = bbox_center
    lxmin, lxmax, lymin, lymax = line_bbox
    return (lxmin <= x_center <= lxmax) and (lymin <= y_center <= lymax)


def process_page(page_id, df):
    detection_root = Path("detection/nomnaocr/labels")
    detection_file = detection_root / f"{page_id}.txt"
    image_root = Path("datasets/nomnaocr/images")

    width, height = imagesize.get(image_root / f"{page_id}.jpg")

    # read label YOLO format
    with open(detection_file, "r") as f:
        labels = f.readlines()
    classes = [label.strip().split()[0] for label in labels]
    bboxes = [list(map(float, label.strip().split()[1:])) for label in labels]
    classes = np.array(classes)
    bboxes = np.array(bboxes)
    bboxes = bboxes * np.array([width, height, width, height])
    bboxes = bboxes.astype(int)

    line_labels_root = Path("datasets/nomnaocr/line_labels")
    line_label_file = line_labels_root / f"{page_id}.txt"
    line_labels = parse_line_labels(line_label_file)

    edit_distances = []
    num_characters = []
    predicted_texts = []
    ground_truth_texts = []

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

    # Process each line and sort bboxes
    for line_idx, line_label in enumerate(line_labels):
        bboxes_ids = line_to_bboxes[line_idx]
        bboxes_ids = sorted(bboxes_ids, key=lambda bid: bbox_center[bid][1])  # sort by y center

        texts = []
        for bbox_id in bboxes_ids:
            row = df[(df["page_id"] == page_id) & (df["bbox_id"] == bbox_id)]
            if not row.empty:
                texts.append(row.iloc[0]["predicted_text"])

        # edit_dist = editdistance.eval("".join(texts), line_label["label"])
        # edit_distances.append(edit_dist)
        # num_characters.append(len(line_label["label"]))
        predicted_texts.append("".join(texts))
        ground_truth_texts.append(line_label["label"])
    page_df = pd.DataFrame({
        "page_id": page_id,
        "predicted_text": predicted_texts,
        "ground_truth_text": ground_truth_texts,
        # "edit_distance": edit_distances,
        # "num_characters": num_characters,
    })
    return page_df


if __name__ == "__main__":
    ocr_csv_path = "ocr_results.csv"
    df = process_ocr_results(ocr_csv_path)
    out_path = Path("evaluation_results")
    page_dfs = []
    for page_id in df["page_id"].unique():
    # for page_id in ['DVSKTT-2 Ngoai ky toan thu/DVSKTT_ngoai_IV_9b']:
        page_out_path = Path(out_path, str(page_id) + ".csv")
        if page_out_path.exists():
            continue

        print(f"Processing page_id: {page_id}")
        page_df = process_page(page_id, df)
        page_dfs.append(page_df)

        page_out_path.parent.mkdir(parents=True, exist_ok=True)
        page_df.to_csv(page_out_path, index=False, encoding="utf-8") 
    # total_edit_distance = final_df["edit_distance"].sum()
    # total_characters = final_df["num_characters"].sum()
    # cer = total_edit_distance / total_characters if total_characters > 0 else 0.0
    # print(f"Character Error Rate (CER): {cer:.4f}")
