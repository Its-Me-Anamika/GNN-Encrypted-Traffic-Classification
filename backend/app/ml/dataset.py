import os
import torch
import numpy as np

try:
    from torch_geometric.data import Data, Dataset
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class DummyDataset:
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

class TrafficFlowDataset:
    """
    Dataset wrapper converting flow feature matrices and 5-tuple relationships into
    PyTorch Geometric Data objects for graph training.
    """
    def __init__(self, node_features, edge_index, labels=None, edge_attr=None):
        self.x = torch.tensor(node_features, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long) if labels is not None else None
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None

        if HAS_PYG:
            self.data = Data(x=self.x, edge_index=self.edge_index, y=self.y, edge_attr=self.edge_attr)
        else:
            self.data = {
                "x": self.x,
                "edge_index": self.edge_index,
                "y": self.y,
                "edge_attr": self.edge_attr
            }

    def get_data(self):
        return self.data

    @classmethod
    def create_split(cls, num_samples=500, num_features=28, num_classes=9, train_ratio=0.7, val_ratio=0.15):
        """Generates synthetic graph split (70/15/15) for dataset bootstrapping."""
        from app.utils.preprocessing import SyntheticDataGenerator

        features_list = []
        labels_list = []
        category_map = {cat: i for i, cat in enumerate(SyntheticDataGenerator.CATEGORIES)}

        for _ in range(num_samples):
            flow = SyntheticDataGenerator.generate_flow()
            features_list.append(flow["features"])
            labels_list.append(category_map[flow["true_category"]])

        x = np.array(features_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.int64)

        # Construct Edge Index based on shared IP indices
        edges = []
        weights = []
        for i in range(num_samples):
            for j in range(i + 1, min(i + 15, num_samples)):
                # Connect nodes with 30% probability or shared synthetic proximity
                if np.random.rand() < 0.35:
                    edges.append([i, j])
                    edges.append([j, i])
                    time_diff = np.random.exponential(scale=2.0)
                    weight = np.exp(-time_diff / 5.0)
                    weights.append([weight])
                    weights.append([weight])

        if not edges:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 1), dtype=np.float32)
        else:
            edge_index = np.array(edges, dtype=np.int64).T
            edge_attr = np.array(weights, dtype=np.float32)

        # Split Indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_mask = np.zeros(num_samples, dtype=bool)
        val_mask = np.zeros(num_samples, dtype=bool)
        test_mask = np.zeros(num_samples, dtype=bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return {
            "x": x,
            "edge_index": edge_index,
            "y": y,
            "edge_attr": edge_attr,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask
        }
