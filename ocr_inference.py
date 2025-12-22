"""Stage 1: Run beam search inference and save outputs."""

from nom_ids_ocr.data import SeqVocab, collate_fn
from nom_ids_ocr.lit_trainer import LitBTTR
import torch
from data import ImageDatasetBBox
from torchvision import transforms
import argparse
from pathlib import Path
from config import Config
from tqdm import tqdm


def load_vocab_and_ids_dict():
    """Load vocabulary and IDS dictionary from files."""
    with open(Config.VOCAB_IDS, 'r', encoding='utf-8') as f:
        base_vocab = f.read().split('\n')
    
    ids_dict = {}
    with open(Config.IDS_EXP, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading IDS dictionary", leave=False):
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


def run_beam_search(model, dataset, output_txt, beam_size=None, max_len=None,
                    alpha=None, device='cuda', process_all=False):
    """Run beam search inference and save outputs to text file."""
    # Use config defaults if not provided
    beam_size = beam_size or Config.OCR_BEAM_SIZE
    max_len = max_len or Config.OCR_MAX_LEN
    alpha = alpha or Config.OCR_ALPHA
    
    # Ensure output directory exists
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_batches = len(dataset) if process_all else 1
    num_outputs = 0
    
    print("\nStage 1: Beam search inference")
    print("-" * 60)
    
    with open(output_path, mode='w', encoding='utf-8') as f:
        with tqdm(total=num_batches, desc="Beam search", unit="batch") as batch_pbar:
            for batch_idx, batch in enumerate(dataset):
                data = collate_fn([batch])
                data.imgs = data.imgs.to(device)
                
                with tqdm(total=len(data.img_bases), desc=f"Batch {batch_idx + 1}", unit="img", leave=False) as img_pbar:
                    for i in range(len(data.img_bases)):
                        with torch.no_grad():
                            output = model.beam_search(
                                data.imgs[i],
                                beam_size=beam_size,
                                max_len=max_len,
                                alpha=alpha
                            )
                        
                        img_id = data.img_bases[i]
                        # Convert output tensor to comma-separated integers
                        output_str = ','.join([str(int(x)) for x in output])
                        f.write(f"{img_id}\t{output_str}\n")
                        num_outputs += 1
                        img_pbar.update(1)
                
                batch_pbar.update(1)
                
                if not process_all:
                    break
    
    print(f"\nSaved {num_outputs} outputs to: {output_path}")


def run_greedy_search(model, dataloader, output_txt, max_len=None,
                      alpha=None, device='cuda', process_all=False):
    """Run greedy search inference in batch mode and save outputs to text file.

    This mirrors run_beam_search but uses `model.greedy_search` on the full
    batch tensor `[B, 3, H, W]` to decode all images in a batch at once.
    """
    # Use config defaults if not provided
    max_len = max_len or Config.OCR_MAX_LEN
    alpha = alpha or Config.OCR_ALPHA

    # Ensure output directory exists
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_batches = len(dataloader) if process_all else 1
    num_outputs = 0

    print("\nStage 1: Greedy search inference (batched)")
    print("-" * 60)

    with open(output_path, mode='w', encoding='utf-8') as f:
        with tqdm(total=num_batches, desc="Greedy search", unit="batch") as batch_pbar:
            for batch_idx, data in enumerate(dataloader):
                # data = collate_fn([batch])
                data.imgs = data.imgs.to(device)

                with torch.no_grad():
                    decoded_seqs = model.greedy_search(
                        data.imgs,
                        max_len=max_len,
                        alpha=alpha,
                    )

                # Write all results for this batch
                with tqdm(total=len(data.img_bases), desc=f"Batch {batch_idx + 1}", unit="img", leave=False) as img_pbar:
                    for i, img_id in enumerate(data.img_bases):
                        seq = decoded_seqs[i]
                        output_str = ','.join([str(int(x)) for x in seq])
                        f.write(f"{img_id}\t{output_str}\n")
                        num_outputs += 1
                        img_pbar.update(1)

                batch_pbar.update(1)

                if not process_all:
                    break

    print(f"\nSaved {num_outputs} outputs to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run OCR beam search inference and save outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-txt",
        dest="output_txt",
        type=str,
        default="inference_outputs.txt",
        help="Path to save the inference outputs text file",
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
        default='cuda',
        help="Device to run inference on (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu')",
    )
    parser.add_argument(
        "--process-all",
        dest="process_all",
        action="store_true",
        help="Process all images in dataset (default: process only first batch)",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=32,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Batch size for data loading",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("OCR Beam Search Inference")
    print("=" * 60)
    print(f"Images directory: {args.image_dir}")
    print(f"Labels directory: {args.label_dir}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output text file: {args.output_txt}")
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
    print("\nCreating dataset...")
    dataset = ImageDatasetBBox(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        vocab=vocab,
        transform=create_transforms(),
        expand_ratio=Config.BBOX_EXPAND_RATIO,
    )

    print(f"Dataset created with {len(dataset)} bboxes images")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Load model
    print("\nLoading model from checkpoint...")
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        vocab_size=len(vocab),
        device=args.device
    )
    print("Model loaded successfully")
    
    # Run beam search
    # run_beam_search(
    #     model=model,
    #     dataset=dataset,
    #     output_txt=args.output_txt,
    #     beam_size=args.beam_size,
    #     max_len=args.max_len,
    #     alpha=args.alpha,
    #     device=args.device,
    #     process_all=args.process_all
    # )

    # Alternatively, run greedy search in batch mode
    run_greedy_search(
        model=model,
        dataloader=dataloader,
        output_txt=args.output_txt,
        max_len=args.max_len,
        alpha=args.alpha,
        device=args.device,
        process_all=args.process_all
    )
    
    print("=" * 60)
    print("Search complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
