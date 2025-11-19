from nom_ids_ocr.data import SeqVocab, collate_fn
from nom_ids_ocr.lit_trainer import LitBTTR
import torch
from data import ImageDataset
from torchvision import transforms
import argparse
import csv
from pathlib import Path
from config import Config


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


def run_inference(model, dataset, output_csv, beam_size=None, max_len=None, 
                  alpha=None, device='cuda', process_all=False):
    """Run OCR inference on dataset and write results to CSV."""
    # Use config defaults if not provided
    beam_size = beam_size or Config.OCR_BEAM_SIZE
    max_len = max_len or Config.OCR_MAX_LEN
    alpha = alpha or Config.OCR_ALPHA
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_text", "predicted_ids"])
        
        total_processed = 0
        for batch_idx, batch in enumerate(dataset):
            data = collate_fn([batch])
            
            for i in range(len(data.img_bases)):
                with torch.no_grad():
                    output = model.beam_search(
                        data.imgs[i].to(device),
                        beam_size=beam_size,
                        max_len=max_len,
                        alpha=alpha
                    )
                
                decoded_text = dataset.vocab.decode(output)
                decoded_ids = ''.join([dataset.vocab.id2char[c] for c in output])
                img_id = data.img_bases[i]
                
                total_processed += 1
                print(f"[{total_processed}] Image ID: {img_id}")
                print(f"  Predicted Text: {decoded_text}")
                print(f"  Predicted IDS: {decoded_ids}")
                
                writer.writerow([img_id, decoded_text, decoded_ids])
            
            if not process_all:
                print(f"\nProcessed first batch only ({total_processed} images)")
                break
        
        if process_all:
            print(f"\nProcessed all images ({total_processed} total)")
    
    print(f"\nResults written to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OCR inference and write results to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default=str(Config.OCR_RESULTS_CSV),
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--images",
        dest="image_dir",
        type=str,
        default=str(Config.IMAGE_ROOT),
        help="Path to images directory",
    )
    parser.add_argument(
        "--labels",
        dest="label_dir",
        type=str,
        default=str(Config.LINE_LABELS_ROOT),
        help="Path to labels directory",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        type=str,
        default=str(Config.OCR_CHECKPOINT),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--beam-size",
        dest="beam_size",
        type=int,
        default=Config.OCR_BEAM_SIZE,
        help="Beam size for beam search",
    )
    parser.add_argument(
        "--max-len",
        dest="max_len",
        type=int,
        default=Config.OCR_MAX_LEN,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha",
        type=float,
        default=Config.OCR_ALPHA,
        help="Length penalty alpha for beam search",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help="Device to run inference on",
    )
    parser.add_argument(
        "--process-all",
        dest="process_all",
        action="store_true",
        help="Process all images in dataset (default: process only first batch)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("OCR Inference Pipeline")
    print("=" * 60)
    print(f"Images directory: {args.image_dir}")
    print(f"Labels directory: {args.label_dir}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output CSV: {args.csv_path}")
    print(f"Device: {args.device}")
    print(f"Beam size: {args.beam_size}")
    print(f"Process all: {args.process_all}")
    print("=" * 60)
    
    # Load vocabulary and IDS dictionary
    print("\nLoading vocabulary and IDS dictionary...")
    base_vocab, ids_dict = load_vocab_and_ids_dict()
    vocab = SeqVocab(base_vocab, ids_dict)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = ImageDataset(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        vocab=vocab,
        transform=create_transforms(),
        expand_ratio=Config.BBOX_EXPAND_RATIO,
    )
    print(f"Dataset created with {len(dataset)} images")
    
    # Load model
    print(f"\nLoading model from checkpoint...")
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        vocab_size=len(vocab),
        device=args.device
    )
    print("Model loaded successfully")
    
    # Run inference
    print(f"\nRunning inference...")
    print("-" * 60)
    run_inference(
        model=model,
        dataset=dataset,
        output_csv=args.csv_path,
        beam_size=args.beam_size,
        max_len=args.max_len,
        alpha=args.alpha,
        device=args.device,
        process_all=args.process_all
    )
    print("=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
