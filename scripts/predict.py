import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tensorflow import keras

from src.losses import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a single image.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--class-names", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=380)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = keras.models.load_model(args.model, custom_objects={"FocalLoss": FocalLoss})
    class_names = Path(args.class_names).read_text().splitlines()

    img = keras.utils.load_img(args.image, target_size=(args.img_size, args.img_size))
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:args.top_k]

    print(f"\nPredictions for {args.image}:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank}. {class_names[idx]:<45} {probs[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
