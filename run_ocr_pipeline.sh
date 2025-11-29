python ocr_inference.py \
  --images datasets/nomnaocr/images \
  --labels datasets/nomnaocr/labels \
  --checkpoint my-models/epoch=248-step=37178-val_ExpRate=0.9878.ckpt \
  --output-txt inference_outputs.txt \
  --process-all

python ocr_decode.py \
  --input-txt inference_outputs.txt \
  --csv evaluation_results/ocr_results_retrain.csv \
  --num-workers 32  