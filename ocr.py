from nom_ids_ocr.data import SeqVocab, collate_fn
from nom_ids_ocr.lit_trainer import LitBTTR
import torch
import pytorch_lightning as pl
from data import ImageDataset
from torchvision import transforms
import argparse
import csv
import os


base_vocab = open('nom-ids/vocab_ids.txt', 'r', encoding='utf-8').read().split('\n')
ids_dict = {
    line.strip().split('\t')[0]: line.strip().split('\t')[1]
    for line in open('nom-ids/ids_exp.txt', 'r', encoding='utf-8').readlines()
}

eval_transforms = transforms.Compose([
        transforms.Resize(size=128),
        transforms.RandomCrop(size=128),
        transforms.RandomInvert(p=1.0),
    ])

parser = argparse.ArgumentParser(description="Run OCR inference and write results to CSV.")
parser.add_argument(
    "--csv",
    dest="csv_path",
    default="ocr_results.csv",
    help="Path to the output CSV file (default: ocr_results.csv)",
)
parser.add_argument(
    "--images",
    dest="image_dir",
    default="datasets/nomnaocr/images",
    help="Path to images directory (default: datasets/nomnaocr/images)",
)
parser.add_argument(
    "--labels",
    dest="label_dir",
    default="datasets/nomnaocr/labels",
    help="Path to labels directory (default: datasets/nomnaocr/labels)",
)
parser.add_argument(
    "--checkpoint",
    dest="checkpoint_path",
    default="my-models/epoch=199-step=19048-val_ExpRate=0.9508.ckpt",
    help="Path to model checkpoint",
)
args = parser.parse_args()

dataset = ImageDataset(
    image_dir=args.image_dir,
    label_dir=args.label_dir,
    vocab=SeqVocab(base_vocab, ids_dict),
    transform=eval_transforms,
    expand_ratio=1.2,
)

model = LitBTTR.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                     map_location='cuda',
    d_model=256, growth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3, beam_size=10, max_len=200, alpha=1.0, learning_rate=1.0, patience=20, vocab_size=len(dataset.vocab), SOS_IDX=1, EOS_IDX=2, PAD_IDX=0)

model.eval()

# Prepare CSV writer
csv_path = args.csv_path
csv_dir = os.path.dirname(csv_path)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)

with open(csv_path, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "predicted_text", "predicted_ids"])  # header

    # Run predictions
    for batch in dataset:
        data = collate_fn([batch])
        for i in range(len(data.img_bases)):
            with torch.no_grad():
                output = model.beam_search(
                    data.imgs[i].to('cuda'), beam_size=3, max_len=200, alpha=1.0
                )
            decoded_output = dataset.vocab.decode(output)
            out_ids = ''.join([dataset.vocab.id2char[c] for c in output])
            img_id = data.img_bases[i]
            print(
                f"Image ID: {img_id}, Predicted Text: {decoded_output}, Predicted IDS: {out_ids}"
            )
            writer.writerow([img_id, decoded_output, out_ids])
        break # Remove this break to process the entire dataset

print(f"Results written to: {csv_path}")
