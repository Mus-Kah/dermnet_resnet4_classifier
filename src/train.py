from pathlib import Path
from typing import Dict, List

from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau,
)

from config import Config
from src.losses import FocalLoss
from src.lr_schedule import cosine_lr_with_warmup

METRICS = lambda: [
    "accuracy",
    keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]


def _phase_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def train_phase1(model, train_ds, test_ds, class_weights: Dict[int, float],
                  cfg: Config) -> keras.callbacks.History:
    _phase_header(f"PHASE 1: Train head with frozen backbone ({cfg.phase1_epochs} epochs)")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.phase1_lr, weight_decay=cfg.weight_decay),
        loss=FocalLoss(gamma=cfg.phase1_focal_gamma, alpha=cfg.phase1_focal_alpha,
                        label_smoothing=cfg.phase1_label_smoothing),
        metrics=METRICS(),
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=cfg.phase1_patience,
                       restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(cfg.save_dir / "phase1_best.keras"),
                         monitor="val_accuracy", save_best_only=True, verbose=1),
        LearningRateScheduler(
            cosine_lr_with_warmup(cfg.phase1_lr, warmup_epochs=cfg.phase1_warmup_epochs,
                                   total_epochs=cfg.phase1_epochs),
            verbose=0,
        ),
    ]

    hist = model.fit(
        train_ds, validation_data=test_ds, epochs=cfg.phase1_epochs,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )
    print(f"Phase 1 best val accuracy: {max(hist.history['val_accuracy']) * 100:.2f}%")
    return hist


def train_phase2(model, base_model, train_ds, test_ds, class_weights: Dict[int, float],
                  cfg: Config, start_epoch: int) -> keras.callbacks.History:
    """Phase 2: unfreeze the top fraction of the backbone and fine-tune."""
    _phase_header(f"PHASE 2: Fine-tune top {int(cfg.phase2_unfreeze_fraction * 100)}% "
                   f"of EfficientNetB4 ({cfg.phase2_epochs} epochs)")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * (1 - cfg.phase2_unfreeze_fraction))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    print(f"   Training {total_layers - freeze_until}/{total_layers} base layers")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.phase2_lr, weight_decay=cfg.weight_decay),
        loss=FocalLoss(gamma=cfg.phase2_focal_gamma, alpha=cfg.phase2_focal_alpha,
                        label_smoothing=cfg.phase2_label_smoothing),
        metrics=METRICS(),
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=cfg.phase2_patience,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.phase2_reduce_lr_patience,
                           min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(cfg.save_dir / "phase2_best.keras"),
                         monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    hist = model.fit(
        train_ds, validation_data=test_ds,
        epochs=start_epoch + cfg.phase2_epochs, initial_epoch=start_epoch,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )
    print(f"Phase 2 best val accuracy: {max(hist.history['val_accuracy']) * 100:.2f}%")
    return hist


def train_phase3(model, base_model, train_ds, test_ds, class_weights: Dict[int, float],
                  cfg: Config, start_epoch: int) -> keras.callbacks.History:
    """Phase 3: unfreeze the entire backbone and fine-tune with a very low LR."""
    _phase_header(f"PHASE 3: Full network fine-tuning ({cfg.phase3_epochs} epochs, very low LR)")

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cfg.phase3_lr,
                                          weight_decay=cfg.weight_decay_phase3),
        loss=FocalLoss(gamma=cfg.phase3_focal_gamma, alpha=cfg.phase3_focal_alpha,
                        label_smoothing=cfg.phase3_label_smoothing),
        metrics=METRICS(),
    )

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=cfg.phase3_patience,
                       restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=cfg.phase3_reduce_lr_patience,
                           min_lr=1e-8, verbose=1),
        ModelCheckpoint(str(cfg.save_dir / "phase3_best.keras"),
                         monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    hist = model.fit(
        train_ds, validation_data=test_ds,
        epochs=start_epoch + cfg.phase3_epochs, initial_epoch=start_epoch,
        class_weight=class_weights, callbacks=callbacks, verbose=1,
    )
    print(f"Phase 3 best val accuracy: {max(hist.history['val_accuracy']) * 100:.2f}%")
    return hist


def run_training_pipeline(model, base_model, train_ds, test_ds,
                           class_weights: Dict[int, float], cfg: Config) -> Dict[str, dict]:
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    all_history: Dict[str, dict] = {}

    hist1 = train_phase1(model, train_ds, test_ds, class_weights, cfg)
    all_history["phase1"] = hist1.history
    start_epoch = len(hist1.history["loss"])

    hist2 = train_phase2(model, base_model, train_ds, test_ds, class_weights, cfg, start_epoch)
    all_history["phase2"] = hist2.history
    start_epoch2 = start_epoch + len(hist2.history["loss"])

    hist3 = train_phase3(model, base_model, train_ds, test_ds, class_weights, cfg, start_epoch2)
    all_history["phase3"] = hist3.history

    return all_history
