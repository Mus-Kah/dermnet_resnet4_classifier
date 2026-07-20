from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing import image_dataset_from_directory

from config import Config

AUTOTUNE = tf.data.AUTOTUNE


def load_raw_datasets(cfg: Config):
    print("\nLoading datasets...")

    raw_train_ds = image_dataset_from_directory(
        cfg.train_dir,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=cfg.seed,
        interpolation="bilinear",
    )

    raw_test_ds = image_dataset_from_directory(
        cfg.test_dir,
        image_size=cfg.img_size,
        batch_size=cfg.batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=cfg.seed,
        interpolation="bilinear",
    )

    class_names = raw_train_ds.class_names
    print(f"Classes ({len(class_names)}): {class_names}")

    return raw_train_ds, raw_test_ds, class_names


def compute_class_weights(raw_train_ds, num_classes: int, cfg: Config) -> Dict[int, float]:
    print("\nComputing class weights...")

    y_labels = np.concatenate(
        [np.argmax(y.numpy(), axis=1) for _, y in raw_train_ds]
    )
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_labels
    )
    class_weights_arr = np.clip(class_weights_arr, cfg.class_weight_min, cfg.class_weight_max)
    class_weights = dict(enumerate(class_weights_arr))

    print(f"   Weight range: [{min(class_weights_arr):.3f}, {max(class_weights_arr):.3f}]")
    return class_weights


def print_class_distribution(y_labels: np.ndarray, class_names: List[str],
                              class_weights: Dict[int, float]) -> None:
    unique, counts = np.unique(y_labels, return_counts=True)
    print("\nClass distribution (training):")
    for i, count in zip(unique, counts):
        print(f"   {class_names[i][:50]:<50} {count:4d} samples "
              f"(weight: {class_weights[i]:.2f})")


def make_augment_fn(img_size: Tuple[int, int]):
    h, w = img_size

    def augment_fn(images, labels):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        images = tf.image.random_brightness(images, max_delta=0.2)
        images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
        images = tf.image.random_saturation(images, lower=0.7, upper=1.3)
        images = tf.image.random_hue(images, max_delta=0.05)

        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        images = tf.image.rot90(images, k=k)

        batch_size = tf.shape(images)[0]
        crop_frac = tf.random.uniform([], minval=0.85, maxval=1.0)
        crop_h = tf.cast(tf.cast(h, tf.float32) * crop_frac, tf.int32)
        crop_w = tf.cast(tf.cast(w, tf.float32) * crop_frac, tf.int32)
        images = tf.image.random_crop(images, size=[batch_size, crop_h, crop_w, 3])
        images = tf.image.resize(images, [h, w])

        if tf.random.uniform([]) > 0.5:
            cutout_size = tf.random.uniform([], minval=50, maxval=100, dtype=tf.int32)
            for i in range(batch_size):
                x = tf.random.uniform([], minval=0, maxval=w - cutout_size, dtype=tf.int32)
                y = tf.random.uniform([], minval=0, maxval=h - cutout_size, dtype=tf.int32)
                mask = tf.ones([cutout_size, cutout_size, 3]) * 128.0
                paddings = [[y, h - y - cutout_size], [x, w - x - cutout_size], [0, 0]]
                mask = tf.pad(mask, paddings)
                images = tf.tensor_scatter_nd_update(
                    images,
                    [[i]],
                    [tf.where(tf.equal(mask, 0.0), images[i], mask)],
                )

        images = tf.clip_by_value(images, 0.0, 255.0)
        return images, labels

    return augment_fn


def build_datasets(cfg: Config):
    raw_train_ds, raw_test_ds, class_names = load_raw_datasets(cfg)
    num_classes = len(class_names)

    y_labels = np.concatenate(
        [np.argmax(y.numpy(), axis=1) for _, y in raw_train_ds]
    )
    class_weights = compute_class_weights(raw_train_ds, num_classes, cfg)
    print_class_distribution(y_labels, class_names, class_weights)

    augment_fn = make_augment_fn(cfg.img_size)
    train_ds = raw_train_ds.map(augment_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = raw_test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, test_ds, class_names, class_weights
