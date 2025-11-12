# Nom-IDS Character Recognition on NomnaOCR Dataset

This project focuses on character recognition for Nom-IDS characters using the NomnaOCR dataset. It involves object detection, OCR inference, and evaluation of results.

## Project Structure

- **`run_pipeline.sh`**: A shell script to run the detection pipeline.
- **`detection.py`**: Detects bounding boxes for Nom-IDS characters using a YOLO model.
- **`ocr.py`**: Performs OCR inference on detected bounding boxes and generates predictions.
- **`eval.py`**: Evaluates the OCR results by comparing predictions with ground truth.

## Dataset

The project uses the **NomnaOCR** dataset, which contains:
- Images of Nom-IDS characters.
- Line-level ground truth labels for evaluation.

## Workflow

1. **Detection**:
   - Use `detection.py` to detect bounding boxes for Nom-IDS characters in the images.
   - Outputs YOLO-format label files.

2. **OCR Inference**:
   - Use `ocr.py` to perform OCR on detected bounding boxes.
   - Outputs a CSV file with predicted text and IDs.

3. **Evaluation**:
   - Use `eval.py` to evaluate OCR predictions against ground truth.
   - Outputs evaluation metrics and per-page results.

## Usage

### 1. Run Detection
```bash
bash run_pipeline.sh
```

### 2. Perform OCR Inference
```bash
python ocr.py --images "datasets/nomnaocr/images" \
              --labels "detection/nomnaocr/labels" \
              --checkpoint "my-models/epoch=199-step=19048-val_ExpRate=0.9508.ckpt" \
              --csv "ocr_results.csv"
```

### 3. Evaluate Results
```bash
python eval.py
```

## Key Components

### Detection
- **Model**: YOLO-based object detection.
- **Input**: Images from `datasets/nomnaocr/images`.
- **Output**: YOLO-format label files in `detection/nomnaocr/labels`.

### OCR
- **Model**: LitBTTR (a transformer-based OCR model).
- **Input**: Detected bounding boxes and images.
- **Output**: CSV file with predicted text and IDs.

### Evaluation
- **Metrics**: Character Error Rate (CER), edit distance, etc.
- **Input**: OCR predictions and ground truth labels.
- **Output**: Evaluation results in `evaluation_results`.

## Dependencies

- Python 3.8+
- PyTorch
- PyTorch Lightning
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Pandas
- tqdm
- imagesize

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ntcuong2103/nom-ids-ocr.git
   cd nom-ids-ocr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Notes

- Ensure the dataset is organized as expected:
  - Images: `datasets/nomnaocr/images`
  - Line labels: `datasets/nomnaocr/line_labels`
- Modify paths in scripts as needed to match your directory structure.

