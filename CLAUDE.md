# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An OCR pipeline for Nôm-IDS characters (Vietnamese classical Chinese-derived script) on the NomnaOCR dataset, plus a
web-based annotation tool for reviewing/correcting predicted labels. The main pipeline lives at the repo root
(Python scripts run via `uv`); `annotator/` is a separate FastAPI + React app for human review of pipeline output.

## Setup

```bash
git clone --branch release https://github.com/ntcuong2103/nom-ids.git nom-ids   # local OCR model package, gitignored
uv sync
```

`nom-ids/` is **not** a git submodule — it's a standalone clone, entirely gitignored, providing the
`nom_ids_ocr` package (`LitBTTR`, `SeqVocab`, `collate_fn`) referenced via `[tool.uv.sources]` (editable path
dependency) in `pyproject.toml`. If `nom-ids/` is missing, pipeline scripts that `import nom_ids_ocr` will fail
before anything else does.

Run all root-level Python scripts through `uv run python <script>.py`, not a bare `python`, so the local
`nom_ids_ocr` editable install resolves correctly.

## Pipeline commands

The pipeline is a strict four-stage chain; each stage's output file is the next stage's input.

```
images/ ──detection.py──► detection/nomnaocr/labels/ (YOLO bbox .txt)
        ──ocr_inference.py──► inference_outputs.txt (raw token sequences, tab-separated)
        ──ocr_decode.py──► ocr_results.csv (predicted_text + predicted_ids)
              ├─ eval.py ──► evaluation_results/ (per-page CSV, CER/edit-distance)
              └─ generate_train_ds.py ──► nomnaocr_labels_v1/ (re-training labels)
```

```bash
# Stage 0 — detection (YOLO)
python detection.py --path "datasets/nomnaocr/images" --model "my-models/best.pt" --output "detection/nomnaocr/labels"

# Stage 1 — OCR inference (greedy is default/fast/batched; beam is per-image and slower)
uv run python ocr_inference.py \
  --images datasets/nomnaocr/images \
  --detection-labels datasets/nomnaocr/detection_labels \
  --checkpoint my-models/epoch=199-step=19048-val_ExpRate=0.9508.ckpt \
  --output-txt inference_outputs.txt \
  --device cuda:0 --batch-size 64 --num-workers 32 --process-all
# omit --process-all to run a quick single-batch smoke test

# Stage 2 — decode token sequences to text/IDS (dedupes identical sequences before decoding)
uv run python ocr_decode.py --input-txt inference_outputs.txt --csv ocr_results.csv --num-workers 8

# Stage 3 — evaluate against ground truth
python eval.py --ocr-csv ocr_results.csv --output-root evaluation_results

# Stage 3b — generate re-training labels (LCS-aligned predicted-vs-ground-truth matches)
python generate_train_ds.py
```

`--help` works on any of the above via `uv run python <script>.py --help`.

There is no test suite, linter, or formatter configured for the root Python codebase.

## Architecture

- **`config.py`** — single `Config` class centralizing all dataset/model/output paths and hyperparameters
  (`YOLO_CONF`, `YOLO_IOU`, `OCR_BEAM_SIZE`, `OCR_MAX_LEN`, `IMAGE_SIZE`, `BBOX_EXPAND_RATIO`, etc). CLI args in the
  pipeline scripts default to these values — check here first when a script's default path/param looks wrong.
- **`data.py`** — `ImageDatasetBBox` (used by the real pipeline) and `ImageDataset` (older/exploratory variant).
  Both crop each YOLO box to a square (expanded by `BBOX_EXPAND_RATIO`, centered on the box) before handing it to
  the OCR model; boxes whose class isn't in `vocab.ids_dict` (and isn't the padding class `'0'`) are dropped.
- **`ocr_inference.py`** — loads `LitBTTR` from a Lightning checkpoint with hardcoded architecture kwargs
  (`d_model=256, growth_rate=24, num_layers=16, nhead=8, num_decoder_layers=3` — must match the checkpoint the model
  was trained with) and writes `image_path@box_id\t<comma-separated token ids>` lines.
- **`ocr_decode.py`** — deduplicates identical token sequences before decoding (many boxes across a page/dataset
  produce the same sequence), decodes each unique sequence once with multiprocessing, then maps results back to
  every image ID. `decoded_text` uses `vocab.decode` (IDS → composed Unicode char where possible); `decoded_ids`
  is the raw IDS string per token.
- **`lcs_util.py`** — longest-common-subsequence alignment, used by `eval.py` and `generate_train_ds.py` to align
  predicted character sequences against ground-truth line text.
- **`eval.py`** — reassembles per-line predictions from per-box outputs (boxes → lines via label metadata), aligns
  to ground truth line-by-line, and writes per-page CSVs for CER computation.
- **`generate_train_ds.py`** — the retraining-data loop: aligns OCR predictions to ground truth via LCS, then
  labels each detected box `correct`/`incorrect` (YOLO-adjacent format: `<character> <x> <y> <w> <h> <selection_flag>`)
  for use as a training signal to improve the recognizer.
- **`nom-ids/`** — the external model package (LitBTTR transformer decoder over a DenseNet encoder), vocab
  (`vocab_ids.txt`), and IDS decomposition dictionary (`ids_exp.txt`, character → IDS string). Vocabulary and IDS
  dict are always loaded together via `load_vocab_and_ids_dict()` in both `ocr_inference.py` and `ocr_decode.py` —
  duplicated rather than shared, so keep both in sync if you change the loading logic.
- **`src/nom_ocr_inference/`** — a stub package entry point (unused placeholder, not part of the real pipeline).

### Annotator (`annotator/`)

A FastAPI backend + Vite/React frontend for reviewing and correcting predicted labels (character + selection
flag) per bounding box, served as one app (backend serves the built frontend as static files + SPA fallback).

- **`backend/settings.py`** — paths are independent of root `config.py` (different label dirs:
  `pseudo_labels_v0.1` / `pseudo_labels_v0.1_edited`, not `detection_labels`). `BBOX_EXPAND_RATIO` is duplicated
  here to match `config.py`'s value — keep them consistent if either changes.
- **`backend/index.py`** — builds an in-memory `pandas.DataFrame` of every box across the whole dataset at startup
  (`BoxIndex`), read from whichever label file exists per page: edited (`LABELS_EDITED_ROOT`) takes priority over
  original (`LABELS_ROOT`). This index is the source of truth the routers query; no database.
- **`backend/store.py`** — edits are copy-on-write: `apply_edit` always rewrites a page's *entire* label file into
  `LABELS_EDITED_ROOT` (write to `.tmp`, then `os.replace` — atomic, crash-safe), never a diff. This is what makes
  "prefer edited file, else original" a correct merge strategy on every reload. The in-memory index is only
  updated after the disk write succeeds, so it never diverges from what's durable.
- **`backend/routers/`** — `volumes.py` (dataset-level summaries), `pages.py` (per-page box CRUD, drives the
  annotation canvas), `crops.py` (paginated/filterable box-crop gallery), `characters.py` (character frequency
  search, autocomplete for corrections).
- **`frontend/src/components/`** — two main views: `PageView/` (canvas overlay for editing boxes on a page image,
  keyboard-shortcut driven via `useReviewShortcuts.ts`) and `Gallery/` (filterable/paginated grid of cropped boxes
  across the dataset, for bulk review by character or confirmation status).

Dev commands (from `annotator/frontend/`):
```bash
npm run dev       # Vite dev server
npm run build     # tsc -b && vite build — outputs to dist/, served by the FastAPI backend
npm run lint       # oxlint
npm run preview
```

Backend dev: `uvicorn annotator.backend.main:app --reload` from the repo root (or `python -m annotator.backend.main`).
Static image serving (`/static/images`) and the built frontend (`/assets`, SPA fallback) are both conditional on
those directories existing — the backend runs fine without a frontend build present, it just won't serve UI routes.

## Gotchas

- `datasets/`, `my-models/`, `nom-ids/`, `detection/`, and `evaluation_results/` are all gitignored — this is a
  large-artifact pipeline where models and datasets are provisioned separately from the checked-in code.
- Two different label formats coexist: YOLO detection format (`<class> <cx> <cy> <w> <h>`, normalized 0–1, used by
  `detection.py`/`ocr_inference.py`) vs. the annotator/training label format (`<character> <x> <y> <w> <h> <selection_flag>`,
  pixel coordinates, used by `generate_train_ds.py` and `annotator/`). Don't assume one script's label directory is
  readable by another without checking its parser.
- OCR model architecture kwargs in `ocr_inference.py`'s `load_model()` must match whatever checkpoint is passed via
  `--checkpoint` — there's no validation that a given `.ckpt` matches the hardcoded dims.
