"""Stage 2: Load inference outputs and decode using multiprocessing."""

from nom_ids_ocr.data import SeqVocab
import argparse
from pathlib import Path
from config import Config
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd


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


_worker_vocab = None


def _init_worker(base_vocab, vocab_dict):
    """Pool initializer: build the SeqVocab (and its Trie) once per worker process."""
    global _worker_vocab
    _worker_vocab = SeqVocab(base_vocab, vocab_dict)


def _decode_output(output_str):
    """Helper function for multiprocessing decoding.

    Args:
        output_str: comma-separated token id string

    Returns:
        tuple of (output_str, decoded_text, decoded_ids)
    """
    vocab = _worker_vocab

    # Convert comma-separated string to list of integers
    output = [int(x) for x in output_str.split(',')]

    decoded_text = ''.join(vocab.decode(output))
    decoded_ids = ''.join([vocab.id2char[c] for c in output])

    return output_str, decoded_text, decoded_ids


def run_decoding(input_txt, output_csv, vocab_dict, base_vocab, num_workers=4):
    """Load inference outputs and decode using multiprocessing.
    
    Optimized to decode only unique output strings, then map results back to image IDs.
    """

    ocr_df = pd.read_csv(input_txt, sep='\t', header=None, names=['image_id', 'output_str'])

    # drop nan
    ocr_df = ocr_df.dropna(subset=['output_str'])

    print(f"Loaded {len(ocr_df)} outputs")
    
    # get unique output strings
    unique_outputs = ocr_df.output_str.unique()

    unique_count = len(unique_outputs)
    dedup_ratio = len(ocr_df) / unique_count if unique_count > 0 else 1
    print(f"Found {unique_count} unique output strings (deduplication ratio: {dedup_ratio:.2f}x)")
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\nStage 2: Parallel decoding")
    print("-" * 60)

    # Decode unique outputs and store results
    decoded_map = {}  # output_str -> (decoded_text, decoded_ids)

    with Pool(num_workers, initializer=_init_worker, initargs=(base_vocab, vocab_dict)) as pool:
        with tqdm(total=len(unique_outputs), desc="Decoding unique strings", unit="str") as pbar:
            for output_str, decoded_text, decoded_ids in pool.imap_unordered(_decode_output, unique_outputs):
                decoded_map[output_str] = (decoded_text, decoded_ids)
                pbar.update(1)
    
    # create columns for decoded results
    ocr_df['predicted_text'] = ocr_df['output_str'].map(lambda x: decoded_map[x][0])
    ocr_df['predicted_ids'] = ocr_df['output_str'].map(lambda x: decoded_map[x][1])
    
    # Write results to CSV
    ocr_df.to_csv(
        output_csv,
        columns=['image_id', 'predicted_text', 'predicted_ids'],
        index=False,
        encoding='utf-8'
    )

    print(f"\nProcessed {len(ocr_df)} images total ({unique_count} unique strings decoded)")
    print(f"Results written to: {output_path}")


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
