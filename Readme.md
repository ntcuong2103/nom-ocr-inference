# Nom-IDS Character Recognition on NomnaOCR Dataset


Goal: OCR pipeline for Nôm-IDS characters (Vietnamese classical script) using the NomnaOCR dataset.

Pipeline:

- Detection — YOLO model detects bounding boxes for Nôm characters in page images
- OCR Inference — LitBTTR (transformer-based) runs on each detected box, outputs a CSV with predicted text + IDs
- Evaluation — Compares predictions against ground truth, computes CER/edit distance



## Project Structure

| File | Description |
|---|---|
| `detection.py` | YOLO-based character bounding box detection |
| `ocr_inference.py` | Stage 1 — greedy/beam search inference; writes raw token sequences to a text file |
| `ocr_decode.py` | Stage 2 — parallel decoding of token sequences into Unicode characters and IDS strings |
| `generate_train_ds.py` | Generates re-training labels by aligning OCR predictions to ground truth via LCS |
| `eval.py` | Evaluates predictions per line against ground truth; outputs per-page CSV results |
| `eval.ipynb` | Interactive evaluation notebook |
| `lcs_util.py` | Longest Common Subsequence utilities for sequence alignment |
| `data.py` | Dataset loader for bounding-box-cropped character images |
| `utils.py` | Shared helpers: bbox parsing, label loading, OCR result preprocessing |
| `config.py` | Centralised path and hyperparameter configuration |
| `run_pipeline.sh` / `run_ocr_pipeline.sh` | Shell scripts to run detection and OCR end-to-end |
| `visualize_yolo.py` | Helper to visualise YOLO detection results |

## Workflow

```
Images
  └─► detection.py          → detection/nomnaocr/labels/  (YOLO bbox files)
        └─► ocr_inference.py → inference_outputs.txt        (raw token sequences)
              └─► ocr_decode.py → ocr_results.csv            (predicted text + IDS)
                    ├─► eval.py              → evaluation_results/  (per-page CSVs)
                    └─► generate_train_ds.py → nomnaocr_labels_v1/  (re-training labels)
```

### Stage 1 — Detection

`detection.py` runs a YOLO model over page images and writes per-page YOLO-format bounding box files.

```bash
bash run_pipeline.sh
```

Key config (`config.py`):
- `YOLO_CONF = 0.1`, `YOLO_IOU = 0.1`, `YOLO_IMGSZ = 1280`, `YOLO_MAX_DET = 500`

### Stage 2 — OCR Inference

`ocr_inference.py` loads the LitBTTR checkpoint, runs **greedy search** (default) or beam search on each detected bounding box, and writes raw token sequences to a tab-separated text file.

```bash
python ocr_inference.py \
  --images datasets/nomnaocr/images \
  --labels detection/nomnaocr/labels \
  --checkpoint my-models/epoch=199-step=19048-val_ExpRate=0.9508.ckpt \
  --output-txt inference_outputs.txt \
  --device cuda:0 \
  --batch-size 64 \
  --num-workers 32 \
  --process-all
```

Key config: `OCR_BEAM_SIZE = 3`, `OCR_MAX_LEN = 63`, `IMAGE_SIZE = 128`, `BBOX_EXPAND_RATIO = 1.2`

### Stage 3 — Decoding

`ocr_decode.py` reads `inference_outputs.txt`, decodes only **unique** token sequences in parallel (deduplication speeds this step significantly), then maps results back to all image IDs and writes `ocr_results.csv`.

```bash
python ocr_decode.py \
  --input-txt inference_outputs.txt \
  --csv ocr_results.csv \
  --num-workers 8
```

### Evaluation

`eval.py` aligns predicted bounding boxes to ground-truth text lines, concatenates per-line predictions, and writes per-page CSV files with predicted and ground-truth text for downstream CER/edit-distance computation.

```bash
python eval.py \
  --ocr-csv ocr_results.csv \
  --output-root evaluation_results
```

### Training Data Generation

`generate_train_ds.py` uses the LCS of predicted and ground-truth text to label each detected character bounding box as a correct match or not. Output labels are used to re-train the IDS recogniser.

```bash
python generate_train_ds.py
```

Output is written to `nomnaocr_labels_v1/` — one text file per page, each line containing `<character> <x> <y> <w> <h> <selection_flag>`.

## Dataset Layout

```
datasets/nomnaocr/
  images/          ← page images (.jpg)
  line_labels/     ← line-level ground truth (.txt, one label per line)

detection/nomnaocr/
  labels/          ← YOLO bounding box files produced by detection.py

nom-ids/
  vocab_ids.txt    ← token vocabulary
  ids_exp.txt      ← IDS expansion dictionary (character → IDS string)

my-models/
  best.pt                                          ← YOLO weights
  epoch=199-step=19048-val_ExpRate=0.9508.ckpt    ← LitBTTR checkpoint
```

## Model

- **Detection**: Ultralytics YOLO
- **OCR**: LitBTTR — a DenseNet encoder + transformer decoder trained on Nôm-IDS sequences (256-dim model, 16 DenseNet layers, 8 attention heads, 3 decoder layers)

## Dependencies

- Python 3.8+
- PyTorch + PyTorch Lightning
- Ultralytics YOLO
- OpenCV, NumPy, Pandas
- tqdm, imagesize
- `nom-ids-ocr` (local package — provides `LitBTTR`, `SeqVocab`, `collate_fn`)

## Installation

```bash
git clone https://github.com/ntcuong2103/nom-ids.git
cd nom-ids
git switch release
cd nom-ids-ocr
pip install -e .
```

## Notes

- Ensure the dataset is organized as expected:
  - Images: `datasets/nomnaocr/images`
  - Line labels: `datasets/nomnaocr/line_labels`
- Modify paths in scripts as needed to match your directory structure.

Recent development trajectory (Dec 2025):

7de18ca → c53cfbb — Initial data loading and OCR scaffolding

e6915f4 / 0ee1d50 — Refactored OCR into a separate pipeline module

2bf6a8c / 5f11785 — Added multiprocessing for dataset generation and evaluation

c03b5da / 89cc9f8 / bec02f4 — Integrated zi-tools IDs, added ID merging, decoded multi-output results

5dab315 — Added greedy search decoding

652c375 — Decodes unique results (dedup logic)

24f417e (HEAD) — Fixed box_id in dataset — likely a bug where box IDs were being assigned incorrectly during training data generation