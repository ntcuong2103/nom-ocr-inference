import glob
import torch
from ultralytics import YOLO
import os
from ultralytics.models.yolo.detect import DetectionPredictor
from tqdm import tqdm

def generate_gt(model_checkpoint, path, output_folder):
    model = YOLO(model_checkpoint)
    # use cuda if available
    if torch.cuda.is_available():
        model.to('cuda')

    for file in tqdm(glob.glob(path + "/**/*.jpg", recursive=True)):
        results = model(file, max_det=500, iou=0.1, conf=0.1, imgsz=1280, single_cls=True, agnostic_nms=True, save=False, show_labels=False, show_conf=False, show_boxes=True)
        id = os.path.relpath(file, path)[:-4]
        for result in results:
            result.save_txt(f"{output_folder}/{id}.txt", save_conf=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="datasets/nomnaocr/val/images")
    parser.add_argument("--model", type=str, default="nom-ocr-project/models/yolo11n-1280-nom-data/weights/best.pt")
    parser.add_argument("--output", type=str, default="nom-detection-gt/nomnaocr-val")
    args = parser.parse_args()

    generate_gt(args.model, args.path, args.output)
