import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.data import build_datasets
from src.evaluate import evaluate_model, save_results
from src.model import build_model_from_config
from src.train import run_training_pipeline
from src.utils import setup_environment


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train the DermNet EfficientNetB4 classifier.")
    parser.add_argument("--data-dir", type=str, default="DermNet",
                         help="Root data directory containing train/ and test/ subfolders")
    parser.add_argument("--save-dir", type=str, default="saved_models/DermNet_EfficientNetB4_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=380, help="Square image size (side length)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--phase1-epochs", type=int, default=50)
    parser.add_argument("--phase2-epochs", type=int, default=50)
    parser.add_argument("--phase3-epochs", type=int, default=50)

    parser.add_argument("--phase1-lr", type=float, default=3e-4)
    parser.add_argument("--phase2-lr", type=float, default=5e-5)
    parser.add_argument("--phase3-lr", type=float, default=1e-5)

    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--no-mixed-precision", action="store_true",
                         help="Disable mixed precision even if a GPU is available")

    args = parser.parse_args()

    return Config(
        data_dir=Path(args.data_dir),
        save_dir=Path(args.save_dir),
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        seed=args.seed,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        phase1_lr=args.phase1_lr,
        phase2_lr=args.phase2_lr,
        phase3_lr=args.phase3_lr,
        dropout_rate=args.dropout_rate,
        use_mixed_precision=not args.no_mixed_precision,
    )


def main() -> None:
    cfg = parse_args()
    setup_environment(cfg.use_mixed_precision)

    train_ds, test_ds, class_names, class_weights = build_datasets(cfg)
    model, base_model = build_model_from_config(len(class_names), cfg)
    model.summary(line_length=100)

    run_training_pipeline(model, base_model, train_ds, test_ds, class_weights, cfg)

    print("\n" + "=" * 80)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    results = evaluate_model(model, test_ds, class_names)
    save_results(model, results, class_names, cfg.save_dir)


if __name__ == "__main__":
    main()
