"""
Unit tests for GNN and Baseline Model Implementations.
Verifies GraphSAGE forward/backward, Mean Aggregation, GAT attention heads,
and Tabular Baselines (Random Forest & MLP).
"""
import unittest
import torch
import numpy as np

from src.models.baseline import TabularBaseline
from src.models.graphsage import SAGEConv, GraphSAGEClassifier, GraphSAGETrainer
from src.models.gat import GATConv, GATClassifier, GATTrainer


class TestModels(unittest.TestCase):
    def setUp(self):
        # 10 nodes, 16 features, 3 classes
        self.num_nodes = 10
        self.in_features = 16
        self.num_classes = 3

        self.x = torch.randn(self.num_nodes, self.in_features)
        # Cycle graph edges: 0-1-2-...-9-0
        src = list(range(self.num_nodes))
        dst = [(i + 1) % self.num_nodes for i in range(self.num_nodes)]
        self.edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        self.y = torch.randint(0, self.num_classes, (self.num_nodes,))

        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.train_mask[:7] = True
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask[7:9] = True
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask[9:] = True

    def test_sage_conv_forward(self):
        conv = SAGEConv(in_features=self.in_features, out_features=8)
        out = conv(self.x, self.edge_index)

        self.assertEqual(out.shape, (self.num_nodes, 8))
        # Check L2 normalization: each row norm should be 1.0
        norms = torch.norm(out, p=2, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=4)

    def test_graphsage_classifier_and_training(self):
        model = GraphSAGEClassifier(
            in_features=self.in_features,
            hidden_dim=16,
            embed_dim=8,
            num_classes=self.num_classes,
        )
        logits = model(self.x, self.edge_index)
        self.assertEqual(logits.shape, (self.num_nodes, self.num_classes))

        trainer = GraphSAGETrainer(model)
        history = trainer.train_graph(
            self.x, self.edge_index, self.y, self.train_mask, self.val_mask, epochs=5
        )
        self.assertEqual(len(history["train_loss"]), 5)

        preds, probs = trainer.predict(self.x, self.edge_index)
        self.assertEqual(len(preds), self.num_nodes)
        self.assertEqual(probs.shape, (self.num_nodes, self.num_classes))

    def test_gat_classifier_and_training(self):
        model = GATClassifier(
            in_features=self.in_features,
            hidden_dim=16,
            embed_dim=8,
            num_classes=self.num_classes,
            num_heads=2,
        )
        logits = model(self.x, self.edge_index)
        self.assertEqual(logits.shape, (self.num_nodes, self.num_classes))

        trainer = GATTrainer(model)
        history = trainer.train_graph(
            self.x, self.edge_index, self.y, self.train_mask, self.val_mask, epochs=5
        )
        self.assertEqual(len(history["train_loss"]), 5)

        preds, probs = trainer.predict(self.x, self.edge_index)
        self.assertEqual(len(preds), self.num_nodes)

    def test_tabular_baselines(self):
        X = self.x.numpy()
        y = self.y.numpy()
        train_idx = self.train_mask.numpy()

        rf = TabularBaseline(model_type="random_forest")
        rf.fit(X[train_idx], y[train_idx])
        rf_preds = rf.predict(X)
        self.assertEqual(len(rf_preds), self.num_nodes)

        mlp = TabularBaseline(model_type="mlp")
        mlp.fit(X[train_idx], y[train_idx])
        mlp_preds = mlp.predict(X)
        self.assertEqual(len(mlp_preds), self.num_nodes)


if __name__ == "__main__":
    unittest.main()
