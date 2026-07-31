import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.dataset import TrafficFlowDataset
from app.ml.trainer import GNNTrainer
from app.ml.baselines import RandomForestBaseline, MLPBaseline
from app.utils.evaluation import compute_classification_metrics
from app.config import settings

def main():
    print("=" * 70)
    print(" MODEL EVALUATION & BASELINE COMPARISON BENCHMARK")
    print("=" * 70)

    # Generate benchmark dataset split
    split = TrafficFlowDataset.create_split(num_samples=700)
    X_train, y_train = split["x"][split["train_mask"]], split["y"][split["train_mask"]]
    X_test, y_test = split["x"][split["test_mask"]], split["y"][split["test_mask"]]

    results = {}

    # 1. GraphSAGE
    print("\n[1/4] Evaluating GraphSAGE GNN...")
    sage_trainer = GNNTrainer(model_type="GraphSAGE", epochs=25)
    results["GraphSAGE"] = sage_trainer.train_graph(split)

    # 2. GAT
    print("\n[2/4] Evaluating GAT GNN...")
    gat_trainer = GNNTrainer(model_type="GAT", epochs=25)
    results["GAT"] = gat_trainer.train_graph(split)

    # 3. Random Forest Baseline
    print("\n[3/4] Evaluating Random Forest Baseline...")
    rf = RandomForestBaseline(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds, _, _ = rf.predict(X_test)
    results["RandomForest"] = compute_classification_metrics(y_test, rf_preds, settings.CLASSES)

    # 4. Results Comparison Table
    print("\n" + "=" * 70)
    print(f"{'MODEL':<20} | {'ACCURACY':<10} | {'MACRO F1':<10} | {'WEIGHTED F1':<12}")
    print("=" * 70)
    for model_name, m in results.items():
        print(f"{model_name:<20} | {m['accuracy']*100:6.2f}%    | {m['macro_f1']*100:6.2f}%    | {m['weighted_f1']*100:8.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
