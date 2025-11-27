from PIL import Image
from nom_ids_ocr.data import SeqVocab, collate_fn
from nom_ids_ocr.lit_trainer import LitBTTR
import torch
from data import ImageDataset
from torchvision import transforms
import argparse
import csv
from pathlib import Path
from config import Config
import imagesize
import numpy as np


def load_vocab_and_ids_dict():
    """Load vocabulary and IDS dictionary from files."""
    with open(Config.VOCAB_IDS, 'r', encoding='utf-8') as f:
        base_vocab = f.read().split('\n')
    
    ids_dict = {}
    with open(Config.IDS_EXP, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ids_dict[parts[0]] = parts[1]
    
    return base_vocab, ids_dict

def create_transforms():
    """Create image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(size=Config.IMAGE_SIZE),
        transforms.RandomCrop(size=Config.IMAGE_SIZE),
        transforms.RandomInvert(p=1.0),
        transforms.ToTensor(), 
    ])

def load_model(checkpoint_path, vocab_size, device='cuda'):
    """Load the trained model from checkpoint."""
    model = LitBTTR.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location=device,
        d_model=256,
        growth_rate=24,
        num_layers=16,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.3,
        beam_size=Config.OCR_BEAM_SIZE,
        max_len=Config.OCR_MAX_LEN,
        alpha=Config.OCR_ALPHA,
        learning_rate=1.0,
        patience=20,
        vocab_size=vocab_size,
        SOS_IDX=1,
        EOS_IDX=2,
        PAD_IDX=0
    )
    model.eval()
    return model

# extract image from bounding box and process
def process_image(image_path, bbox, expand_ratio=1.2, transform=None):
    image = Image.open(image_path).convert('RGB')
    x, y, w_bbox, h_bbox = bbox
    
    # make equal crop
    w = max(w_bbox, h_bbox) * expand_ratio
    h = w

    # calculate coordinates
    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(image.width, int(x + w / 2))
    y2 = min(image.height, int(y + h / 2))

    # crop the image
    image_cropped = image.crop((x1, y1, x2, y2))

    # check size of the cropped image
    if image_cropped.size[0] < 1 or image_cropped.size[1] < 1:
        print(f"Invalid crop size for {image_path} at index {idx}. Skipping.")
        return None
    if transform:
        image_cropped = transform(image_cropped)
    return image_cropped

    
def infer_ocr(model, image_cropped, vocab, device, beam_size, max_len, alpha):
    with torch.no_grad():
        output = model.beam_search(
            image_cropped.to(device),
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha
        )
    
    decoded_text = vocab.decode(output)
    decoded_ids = ''.join([vocab.id2char[c] for c in output])
    return decoded_text, decoded_ids

# API: image, list <bounding boxes> -> ocr results (list of texts)
# API: image, bounding box -> ocr result (text)

def main():
    parser = argparse.ArgumentParser(description="OCR Inference Script")
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint', default=Config.OCR_CHECKPOINT)
    args = parser.parse_args()

    base_vocab, ids_dict = load_vocab_and_ids_dict()
    vocab = SeqVocab(base_vocab, ids_dict)
    transform = create_transforms()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, vocab_size=len(vocab), device=device)
    model.to(device)

    # read bounding boxes from file
    bbox_file = Path("detection/nomnaocr/labels/DVSKTT-1 Quyen thu/DVSKTT_thu_I_1a.txt")
    image_file = Path("datasets/nomnaocr/images/DVSKTT-1 Quyen thu/DVSKTT_thu_I_1a.jpg")
    bboxes = []
    image_size = imagesize.get(image_file)

    with open(bbox_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bboxes.append(tuple(map(float, parts[1:5])))
    bboxes = np.array(bboxes) * np.array([image_size[0], image_size[1], image_size[0], image_size[1]])

    # crop first bounding box and run OCR    

    image_cropped = process_image(image_file, bboxes[0], transform=transform)

    ocr_text, ocr_ids = infer_ocr(
        model,
        image_cropped,
        vocab,
        device,
        beam_size=Config.OCR_BEAM_SIZE,
        max_len=Config.OCR_MAX_LEN,
        alpha=Config.OCR_ALPHA
    )
    pass

    #     row['ocr_text'] = ocr_text
    #     row['ocr_ids'] = ocr_ids
    #     writer.writerow(row)

if __name__ == "__main__":
    main()