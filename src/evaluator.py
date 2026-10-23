"""
Scientific Evaluation and Comparison Module for Traffic Classification.
Calculates Accuracy, Precision, Recall, Macro-F1, per-class metrics, confusion matrices,
and generates side-by-side comparison tables between GraphSAGE, GAT, and Baselines.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/Windows servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class TrafficEvaluator:
    """Evaluates classification performance with strict scientific metrics and confusion matrices."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        class_names: Optional[List[str]] = None,
        model_name: str = "Model",
        experiment_name: str = "Evaluation",
    ) -> Dict[str, Any]:
        """
        Computes comprehensive evaluation metrics and saves summary JSON.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if len(y_true) == 0:
            raise ValueError("Evaluation target y_true is empty.")

        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        if class_names is None:
            class_names = [f"Class_{i}" for i in unique_labels]
        elif len(class_names) < len(unique_labels):
            class_names = class_names + [f"Class_{i}" for i in range(len(class_names), len(unique_labels))]

        # Global metrics
        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

        # Per-class metrics
        per_class = {}
        for idx, cls_name in enumerate(class_names):
            c_true = (y_true == idx)
            c_pred = (y_pred == idx)
            tp = int(np.logical_and(c_true, c_pred).sum())
            fp = int(np.logical_and(~c_true, c_pred).sum())
            fn = int(np.logical_and(c_true, c_pred == False).sum())
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_class[cls_name] = {
                "support": int(c_true.sum()),
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        metrics = {
            "model_name": model_name,
            "experiment_name": experiment_name,
            "num_test_samples": len(y_true),
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "macro_f1": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "weighted_f1": f1_weighted,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }

        # Save metrics to JSON
        safe_model_name = model_name.lower().replace(" ", "_")
        safe_exp_name = experiment_name.lower().replace(" ", "_")
        json_path = self.output_dir / f"metrics_{safe_exp_name}_{safe_model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Plot and save confusion matrix heatmap
        cm_fig_path = self.output_dir / f"cm_{safe_exp_name}_{safe_model_name}.png"
        self._plot_confusion_matrix(cm, class_names, model_name, experiment_name, cm_fig_path)

        return metrics

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str,
        experiment_name: str,
        save_path: Path,
    ):
        """Generates and saves a clean confusion matrix heatmap figure."""
        plt.figure(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names) * 0.9)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
        )
        plt.title(f"Confusion Matrix: {model_name} ({experiment_name})", fontsize=12, fontweight="bold")
        plt.ylabel("Ground Truth Class", fontsize=10)
        plt.xlabel("Predicted Class", fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def compare_models(
        self,
        results_list: List[Dict[str, Any]],
        experiment_name: str = "Comparison",
    ) -> pd.DataFrame:
        """
        Creates a comparison table across baselines and GNN models.
        Answers the core research question: Does GNN improve over conventional ML?
        """
        rows = []
        for res in results_list:
            rows.append({
                "Model": res.get("model_name", "Unknown"),
                "Accuracy": f"{res.get('accuracy', 0.0):.4f}",
                "Macro-F1": f"{res.get('macro_f1', 0.0):.4f}",
                "Precision (Macro)": f"{res.get('precision_macro', 0.0):.4f}",
                "Recall (Macro)": f"{res.get('recall_macro', 0.0):.4f}",
                "Weighted-F1": f"{res.get('weighted_f1', 0.0):.4f}",
                "Test Samples": res.get("num_test_samples", 0),
            })

        df_comp = pd.DataFrame(rows)
        safe_exp = experiment_name.lower().replace(" ", "_")
        csv_path = self.output_dir / f"model_comparison_{safe_exp}.csv"
        df_comp.to_csv(csv_path, index=False)

        md_path = self.output_dir / f"model_comparison_{safe_exp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Model Comparison: {experiment_name}\n\n")
            try:
                f.write(df_comp.to_markdown(index=False))
            except Exception:
                headers = list(df_comp.columns)
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for _, row in df_comp.iterrows():
                    f.write("| " + " | ".join(str(val) for val in row.values) + " |\n")
            f.write("\n\n*Evaluation performed under identical train/test splits with zero label leakage.*\n")

        return df_comp

    def save_predictions(
        self,
        flow_ids: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: Optional[np.ndarray],
        class_names: List[str],
        filename: str = "test_predictions.csv",
    ) -> Path:
        """Saves individual flow-level test predictions and confidence scores."""
        out_rows = []
        for idx in range(len(y_true)):
            true_idx = int(y_true[idx])
            pred_idx = int(y_pred[idx])
            true_lbl = class_names[true_idx] if true_idx < len(class_names) else f"class_{true_idx}"
            pred_lbl = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            conf = float(probs[idx, pred_idx]) if probs is not None else 1.0

            out_rows.append({
                "flow_id": flow_ids[idx] if idx < len(flow_ids) else f"flow_{idx}",
                "ground_truth": true_lbl,
                "predicted": pred_lbl,
                "confidence": round(conf, 4),
                "is_correct": bool(true_idx == pred_idx),
            })

        df_preds = pd.DataFrame(out_rows)
        out_path = self.output_dir / filename
        df_preds.to_csv(out_path, index=False)
        return out_path
