import os
import sys
import torch
import numpy as np

# Add app to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.ml.dataset import TrafficFlowDataset
from app.ml.trainer import GNNTrainer
from app.models.database import init_db, SessionLocal, TrafficFlow, ClassificationMetric
from app.utils.evaluation import compute_classification_metrics

def main():
    print("=" * 70)
    print(" GNN ENCRYPTED TRAFFIC CLASSIFICATION - SYNTHETIC DATA BOOTSTRAPPER")
    print("=" * 70)

    # 1. Initialize Database
    print("\n[1/3] Initializing SQLite Database...")
    init_db()

    # 2. Train and Save GNN Model Checkpoint
    print("\n[2/3] Generating Synthetic Flow Dataset & Training GNN Model...")
    split_data = TrafficFlowDataset.create_split(
        num_samples=600,
        num_features=settings.NUM_FEATURES,
        num_classes=len(settings.CLASSES),
        train_ratio=0.7,
        val_ratio=0.15
    )

    trainer = GNNTrainer(model_type="GraphSAGE", lr=0.002, epochs=30)
    metrics = trainer.train_graph(split_data)

    model_dir = os.path.dirname(settings.MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_checkpoint(settings.MODEL_PATH)

    print(f"\n[+] GNN Model Trained Successfully!")
    print(f"    - Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro F1:    {metrics['macro_f1'] * 100:.2f}%")
    print(f"    - Weighted F1: {metrics['weighted_f1'] * 100:.2f}%")

    # 3. Save Performance Metrics in DB
    print("\n[3/3] Saving Evaluation Metrics to Database...")
    db = SessionLocal()
    try:
        db.query(ClassificationMetric).delete()
        metric_record = ClassificationMetric(
            model_name="GraphSAGE (PyTorch Geometric)",
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
            weighted_f1=metrics["weighted_f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            per_class_metrics=metrics["per_class_metrics"],
            confusion_matrix=metrics["confusion_matrix"]
        )
        db.add(metric_record)
        db.commit()
        print("[+] Saved metrics record to SQLite database.")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print(" BOOTSTRAP COMPLETE! You can now start the FastAPI backend.")
    print("=" * 70)

if __name__ == "__main__":
    main()
