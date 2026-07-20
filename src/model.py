from typing import Tuple

from tensorflow import keras
from tensorflow.keras import applications, layers, regularizers

from config import Config


def build_efficientnet_model(num_classes: int, img_size: Tuple[int, int],
                              l2_reg: float = 1e-4, dropout_rate: float = 0.5,
                              model_name: str = "DermNet_EfficientNetB4_v2"):
    inputs = keras.Input(shape=(*img_size, 3), name="input_images")

    base_model = applications.EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs,
        pooling=None,
    )
    base_model.trainable = False

    x = base_model.output

    # Global Average Pooling + Global Max Pooling (richer features)
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([gap, gmp])

    # Dense layers with strong regularization
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dense(
        512, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg), name="dense1",
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg), name="dense2",
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="drop2")(x)

    # Output layer (float32 for numerical stability under mixed precision)
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, base_model


def build_model_from_config(num_classes: int, cfg: Config):
    return build_efficientnet_model(
        num_classes=num_classes,
        img_size=cfg.img_size,
        l2_reg=cfg.l2_reg,
        dropout_rate=cfg.dropout_rate,
        model_name=cfg.model_name,
    )
