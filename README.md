# DermNet EfficientNetB4 Classifier

A structured, script-based rewrite of the `Light_dermnet_autoencoder_pipeline.ipynb`
notebook: a 23-class skin-disease classification pipeline built on
EfficientNetB4, focal loss, strong on-the-fly augmentation, and progressive
3-phase fine-tuning. No frontend/serving component — this is training +
evaluation + inference code only.

## Project layout

```
dermnet_classifier/
├── config.py            # Config dataclass — every hyperparameter lives here
├── requirements.txt
├── scripts/
│   ├── train.py          # CLI: run the full training pipeline
│   ├── evaluate.py        # CLI: evaluate a saved model on a test set
│   └── predict.py         # CLI: classify a single image
└── src/
    ├── data.py            # dataset loading, class weights, augmentation
    ├── model.py           # EfficientNetB4 architecture
    ├── losses.py          # FocalLoss
    ├── lr_schedule.py     # cosine-with-warmup LR schedule
    ├── train.py            # 3-phase training pipeline (frozen → partial → full fine-tune)
    ├── evaluate.py          # metrics, classification report, save_results
    └── utils.py             # TF/GPU/mixed-precision setup
```

## Expected data layout

```
DermNet/
├── train/
│   ├── Acne and Rosacea Photos/
│   ├── Eczema Photos/
│   └── ... (23 class folders)
└── test/
    ├── Acne and Rosacea Photos/
    └── ...
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py \
    --data-dir DermNet \
    --save-dir saved_models/DermNet_EfficientNetB4_v2 \
    --phase1-epochs 50 --phase2-epochs 50 --phase3-epochs 50
```

Key flags (see `python scripts/train.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `DermNet` | Root folder containing `train/` and `test/` |
| `--batch-size` | `16` | Batch size |
| `--img-size` | `380` | Square input resolution (EfficientNetB4 native) |
| `--phase1/2/3-epochs` | `50` each | Epochs per fine-tuning phase |
| `--phase1/2/3-lr` | `3e-4 / 5e-5 / 1e-5` | Learning rate per phase |
| `--dropout-rate` | `0.5` | Dropout in the classification head |
| `--no-mixed-precision` | off | Disable float16 mixed precision |

All other tunables (focal loss gamma/alpha, label smoothing, weight decay,
patience values, unfreeze fraction, class-weight caps) can be changed by
editing `config.py` or passing a custom `Config(...)` in your own script.

### Training pipeline

1. **Phase 1** — train the classification head only, backbone frozen.
2. **Phase 2** — unfreeze the top 40% of EfficientNetB4 layers and fine-tune
   at a lower LR.
3. **Phase 3** — unfreeze the entire backbone and fine-tune at a very low LR.

Each phase uses `AdamW`, a `FocalLoss` (with per-phase gamma/alpha/label
smoothing), early stopping on `val_accuracy`, and checkpoints the best model
to `<save-dir>/phaseN_best.keras`.

## Outputs

After training, `<save-dir>/` contains:

- `phase1_best.keras`, `phase2_best.keras`, `phase3_best.keras` — best
  checkpoint per phase
- `<model_name>_final.keras` — final model after all phases
- `class_names.txt` — class index → label mapping used at inference time
- `per_class_metrics.csv` — precision/recall/F1/support per class
- `summary_metrics.csv` — top-1/top-3/top-5 accuracy and weighted F1

## Evaluating a saved model

```bash
python scripts/evaluate.py \
    --model saved_models/DermNet_EfficientNetB4_v2/DermNet_EfficientNetB4_v2_final.keras \
    --test-dir DermNet/test \
    --class-names saved_models/DermNet_EfficientNetB4_v2/class_names.txt
```

## Predicting on a single image

```bash
python scripts/predict.py \
    --model saved_models/DermNet_EfficientNetB4_v2/DermNet_EfficientNetB4_v2_final.keras \
    --class-names saved_models/DermNet_EfficientNetB4_v2/class_names.txt \
    --image path/to/some_lesion.jpg \
    --top-k 3
```
