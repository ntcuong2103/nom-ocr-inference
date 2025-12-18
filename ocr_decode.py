"""Stage 2: Load inference outputs and decode using multiprocessing."""

from nom_ids_ocr.data import SeqVocab
import argparse
import csv
from pathlib import Path
from config import Config
from tqdm import tqdm
from multiprocessing import Pool


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


def _decode_output(args):
    """Helper function for multiprocessing decoding.
    
    Args:
        args: tuple of (img_id, output_str, vocab_dict, base_vocab)
    
    Returns:
        tuple of (img_id, decoded_text, decoded_ids)
    """
    img_id, output_str, vocab_dict, base_vocab = args
    
    # Recreate SeqVocab instance for this process
    vocab = SeqVocab(base_vocab, vocab_dict)
    
    # Convert comma-separated string to list of integers
    output = [int(x) for x in output_str.split(',')]
    
    decoded_text = ''.join(vocab.decode(output))
    decoded_ids = ''.join([vocab.id2char[c] for c in output])
    
    return img_id, decoded_text, decoded_ids


def run_decoding(input_txt, output_csv, vocab_dict, base_vocab, num_workers=4):
    """Load inference outputs and decode using multiprocessing."""
    # Load outputs from text file
    print(f"\nLoading inference outputs from {input_txt}...")
    outputs_data = []
    with open(input_txt, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_id, output_str = parts
                outputs_data.append((img_id, output_str))
    
    print(f"Loaded {len(outputs_data)} outputs")
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStage 2: Parallel decoding")
    print("-" * 60)
    
    # Prepare arguments for multiprocessing
    decode_args = [
        (img_id, output_str, vocab_dict, base_vocab)
        for img_id, output_str in outputs_data
    ]
    
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "predicted_text", "predicted_ids"])
        
        # Use multiprocessing pool for decoding
        with Pool(num_workers) as pool:
            with tqdm(total=len(decode_args), desc="Decoding", unit="img") as pbar:
                for img_id, decoded_text, decoded_ids in pool.imap_unordered(_decode_output, decode_args):
                    writer.writerow([img_id, decoded_text, decoded_ids])
                    pbar.update(1)
    
    print(f"\nProcessed {len(outputs_data)} images total")
    print(f"\nResults written to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode OCR inference outputs using multiprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-txt",
        dest="input_txt",
        type=str,
        default="inference_outputs.txt",
        help="Path to the inference outputs text file",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default=str(Config.OCR_RESULTS_CSV),
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=8,
        help="Number of workers for multiprocessing decoding",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("OCR Parallel Decoding")
    print("=" * 60)
    print(f"Input text file: {args.input_txt}")
    print(f"Output CSV: {args.csv_path}")
    print(f"Number of workers: {args.num_workers}")
    print("=" * 60)
    
    # Load vocabulary and IDS dictionary
    print("\nLoading vocabulary and IDS dictionary...")
    base_vocab, ids_dict = load_vocab_and_ids_dict()
    print(f"Vocabulary size: {len(base_vocab)}")
    
    # Run decoding
    run_decoding(
        input_txt=args.input_txt,
        output_csv=args.csv_path,
        vocab_dict=ids_dict,
        base_vocab=base_vocab,
        num_workers=args.num_workers
    )
    
    print("=" * 60)
    print("Decoding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
