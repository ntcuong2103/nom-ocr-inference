"""Configuration settings for OCR pipeline."""

from pathlib import Path


class Config:
    """Configuration class for OCR pipeline."""
    
    # Dataset paths
    DATASET_ROOT = Path("datasets/nomnaocr")
    IMAGE_ROOT = DATASET_ROOT / "images"
    LINE_LABELS_ROOT = DATASET_ROOT / "line_labels"
    
    # Detection paths
    DETECTION_LABELS = DATASET_ROOT / "detection_labels"
    
    # Model paths
    MODELS_ROOT = Path("my-models")
    YOLO_MODEL = MODELS_ROOT / "best.pt"
    OCR_CHECKPOINT = MODELS_ROOT / "epoch=199-step=19048-val_ExpRate=0.9508.ckpt"
    
    # Vocabulary paths
    VOCAB_IDS = Path("nom-ids/vocab_ids.txt")
    IDS_EXP = Path("nom-ids/ids_exp.txt")
    
    # Output paths
    OUTPUT_ROOT = Path("evaluation_results")
    OCR_RESULTS_CSV = Path("ocr_results.csv")
    
    # Model parameters
    YOLO_MAX_DET = 500
    YOLO_IOU = 0.1
    YOLO_CONF = 0.1
    YOLO_IMGSZ = 1280
    
    OCR_BEAM_SIZE = 3
    OCR_MAX_LEN = 63
    OCR_ALPHA = 1.0
    
    # Image processing
    IMAGE_SIZE = 128
    BBOX_EXPAND_RATIO = 1.2
