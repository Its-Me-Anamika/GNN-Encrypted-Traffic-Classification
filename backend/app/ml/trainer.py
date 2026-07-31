import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from app.ml.models import EncryptedTrafficGNN
from app.utils.evaluation import compute_classification_metrics
from app.config import settings

class GNNTrainer:
    """Training pipeline for GNN Encrypted Traffic Classifier."""
    
    def __init__(self, model_type="GraphSAGE", lr=0.001, weight_decay=1e-4, epochs=50):
        self.model_type = model_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = EncryptedTrafficGNN(
            architecture=model_type,
            in_channels=settings.NUM_FEATURES,
            out_channels=len(settings.CLASSES),
            dropout_rate=0.3
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train_graph(self, graph_data):
        """
        Trains model on single graph data dictionary / PyG Data object.
        """
        if isinstance(graph_data, dict):
            x = torch.tensor(graph_data["x"], dtype=torch.float32).to(self.device)
            edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long).to(self.device)
            y = torch.tensor(graph_data["y"], dtype=torch.long).to(self.device)
            train_mask = torch.tensor(graph_data["train_mask"], dtype=torch.bool).to(self.device)
            val_mask = torch.tensor(graph_data["val_mask"], dtype=torch.bool).to(self.device)
            test_mask = torch.tensor(graph_data["test_mask"], dtype=torch.bool).to(self.device)
        else:
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)
            y = graph_data.y.to(self.device)
            train_mask = graph_data.train_mask.to(self.device)
            val_mask = graph_data.val_mask.to(self.device)
            test_mask = graph_data.test_mask.to(self.device)

        best_val_acc = 0.0
        best_model_weights = None

        print(f"[*] Starting GNN ({self.model_type}) training for {self.epochs} epochs on device: {self.device}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(x, edge_index)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            self.optimizer.step()

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x, edge_index)
                val_preds = torch.argmax(val_logits[val_mask], dim=-1)
                val_acc = float((val_preds == y[val_mask]).sum() / len(y[val_mask]))

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_model_weights = self.model.state_dict()

            if epoch % 10 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:03d}/{self.epochs:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

        # Load best weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        # Test evaluation
        self.model.eval()
        with torch.no_grad():
            test_logits = self.model(x, edge_index)
            test_preds = torch.argmax(test_logits[test_mask], dim=-1).cpu().numpy()
            y_test = y[test_mask].cpu().numpy()

        metrics = compute_classification_metrics(y_test, test_preds, settings.CLASSES)
        return metrics

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_type": self.model_type,
            "classes": settings.CLASSES,
            "num_features": settings.NUM_FEATURES
        }, filepath)
        print(f"[+] Model checkpoint saved to: {filepath}")

    def load_checkpoint(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print(f"[+] Successfully loaded model checkpoint from {filepath}")
            return True
        return False
