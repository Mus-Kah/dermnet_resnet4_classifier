from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from tensorflow.keras.utils import to_categorical


def evaluate_model(model, dataset, class_names: List[str]) -> Dict:
    y_true_list, y_pred_probs_list = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs_list.extend(probs)

    y_true = np.array(y_true_list)
    y_pred_probs = np.array(y_pred_probs_list)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    top3 = np.mean(np.any(np.argsort(y_pred_probs, axis=1)[:, -3:] == y_true[:, None], axis=1))
    top5 = np.mean(np.any(np.argsort(y_pred_probs, axis=1)[:, -5:] == y_true[:, None], axis=1))

    try:
        auc = roc_auc_score(
            to_categorical(y_true, num_classes=len(class_names)),
            y_pred_probs, average="weighted", multi_class="ovr",
        )
    except ValueError:
        auc = None

    print(f"\n  Top-1 Accuracy : {acc * 100:.2f}%")
    print(f"  Top-3 Accuracy : {top3 * 100:.2f}%")
    print(f"  Top-5 Accuracy : {top5 * 100:.2f}%")
    print(f"  Precision (W)  : {prec:.4f}")
    print(f"  Recall (W)     : {rec:.4f}")
    print(f"  F1-Score (W)   : {f1:.4f}")
    if auc is not None:
        print(f"  AUC-ROC (W)    : {auc:.4f}")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).T

    print("\n── Per-Class F1 Scores ──")
    per_class_f1 = df_report.loc[class_names, "f1-score"].sort_values()
    for cls, score in per_class_f1.items():
        support = df_report.loc[cls, "support"]
        print(f"  {cls[:55]:<55} F1: {score:.4f} (n={int(support)})")

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc, "top3": top3, "top5": top5,
        "precision": prec, "recall": rec, "f1": f1, "auc": auc,
        "cm": cm, "report": report_dict, "df_report": df_report,
    }


def save_results(model, results: Dict, class_names: List[str], save_dir: Path) -> None:
    """Persist the trained model, class names, and evaluation reports to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving final model...")
    final_model_path = save_dir / f"{model.name}_final.keras"
    model.save(str(final_model_path))
    print(f"Saved: {final_model_path}")

    class_names_file = save_dir / "class_names.txt"
    with open(class_names_file, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class names: {class_names_file}")

    results["df_report"].round(4).to_csv(save_dir / "per_class_metrics.csv")

    summary = {
        "top1_accuracy": f"{results['accuracy'] * 100:.2f}%",
        "top3_accuracy": f"{results['top3'] * 100:.2f}%",
        "top5_accuracy": f"{results['top5'] * 100:.2f}%",
        "weighted_f1": f"{results['f1']:.4f}",
    }
    pd.DataFrame([summary]).to_csv(save_dir / "summary_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("🏆 TRAINING COMPLETE")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k:<20} {v}")
    print("=" * 80)
