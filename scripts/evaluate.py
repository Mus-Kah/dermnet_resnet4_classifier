import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from src.evaluate import evaluate_model, save_results
from src.losses import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved DermNet classifier.")
    parser.add_argument("--model", type=str, required=True, help="Path to a .keras model file")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--class-names", type=str, required=True,
                         help="Path to a newline-delimited class_names.txt file")
    parser.add_argument("--img-size", type=int, default=380)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default=None,
                         help="If set, write metrics CSVs here (defaults to the model's folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = keras.models.load_model(args.model, custom_objects={"FocalLoss": FocalLoss})

    class_names = Path(args.class_names).read_text().splitlines()

    test_ds = image_dataset_from_directory(
        args.test_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
        interpolation="bilinear",
    ).cache().prefetch(1)

    results = evaluate_model(model, test_ds, class_names)

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.model).parent
    save_results(model, results, class_names, save_dir)


if __name__ == "__main__":
    main()
