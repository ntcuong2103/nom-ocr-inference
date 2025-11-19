"""YOLO-based object detection for Nom-IDS characters."""

import argparse
import glob
import logging
from pathlib import Path

import torch
from ultralytics import YOLO
from tqdm import tqdm

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_detections(model_checkpoint: str, image_path: str, output_folder: str) -> None:
    """Generate bounding box detections using YOLO model.
    
    Args:
        model_checkpoint: Path to YOLO model checkpoint
        image_path: Path to images directory
        output_folder: Path to output labels directory
    """
    model = YOLO(model_checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Using device: {device}")
    
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(glob.glob(f"{image_path}/**/*.jpg", recursive=True))
    logger.info(f"Found {len(image_files)} images to process")

    for file in tqdm(image_files, desc="Processing images"):
        results = model(
            file,
            max_det=Config.YOLO_MAX_DET,
            iou=Config.YOLO_IOU,
            conf=Config.YOLO_CONF,
            imgsz=Config.YOLO_IMGSZ,
            single_cls=True,
            agnostic_nms=True,
            save=False,
            show_labels=False,
            show_conf=False,
            show_boxes=True
        )
        
        relative_path = Path(file).relative_to(image_path).with_suffix('')
        output_file = output_path / f"{relative_path}.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            result.save_txt(str(output_file), save_conf=False)


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on images")
    parser.add_argument("--path", type=str, default=str(Config.IMAGE_ROOT),
                        help="Path to images directory")
    parser.add_argument("--model", type=str, default=str(Config.YOLO_MODEL),
                        help="Path to YOLO model checkpoint")
    parser.add_argument("--output", type=str, default=str(Config.DETECTION_LABELS),
                        help="Path to output labels directory")
    args = parser.parse_args()

    generate_detections(args.model, args.path, args.output)
    logger.info("Detection completed successfully")


if __name__ == "__main__":
    main()
