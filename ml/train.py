import os
import sys
import argparse
import yaml
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.dataset import TrafficFlowDataset
from app.ml.trainer import GNNTrainer
from app.config import settings

def main():
    parser = argparse.ArgumentParser(description="Train GNN Encrypted Traffic Classifier")
    parser.add_argument("--model", type=str, default="GraphSAGE", choices=["GraphSAGE", "GAT"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples", type=int, default=800, help="Number of graph nodes/samples")
    args = parser.parse_args()

    print(f"[*] Training {args.model} GNN Model ({args.epochs} Epochs, LR={args.lr})...")

    graph_data = TrafficFlowDataset.create_split(
        num_samples=args.samples,
        num_features=28,
        num_classes=len(settings.CLASSES),
        train_ratio=0.7,
        val_ratio=0.15
    )

    trainer = GNNTrainer(model_type=args.model, lr=args.lr, epochs=args.epochs)
    metrics = trainer.train_graph(graph_data)

    save_path = os.path.join("backend", "app", "ml", "saved_models", f"gnn_{args.model.lower()}_model.pt")
    trainer.save_checkpoint(save_path)

    print("\n[+] Training Complete!")
    print(f"    - Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro F1:    {metrics['macro_f1'] * 100:.2f}%")
    print(f"    - Saved Model: {save_path}")

if __name__ == "__main__":
    main()
