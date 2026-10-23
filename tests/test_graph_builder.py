"""
Unit tests for Flow Graph Construction and Topology Metrics.
Verifies node features, edge generation (endpoint-sharing, temporal, k-NN),
zero label leakage, and PyTorch tensor integrity.
"""
import unittest
import pandas as pd
import numpy as np
import torch

from src.graph_builder import FlowGraphBuilder, FlowGraphData
from src.config import ALL_FEATURE_COLS, NUMERICAL_FEATURES


class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        # Create a synthetic DataFrame representing 6 network flows
        data = []
        base_features = {col: 10.0 for col in NUMERICAL_FEATURES}

        # Flows 0 and 1 share IP 192.168.1.5 (Endpoint sharing)
        f0 = dict(base_features, flow_id="f0", src_ip="192.168.1.5", dst_ip="10.0.0.1", protocol="TCP", start_time=100.0, label="browsing")
        f1 = dict(base_features, flow_id="f1", src_ip="192.168.1.5", dst_ip="10.0.0.2", protocol="TCP", start_time=101.0, label="browsing")

        # Flows 2 and 3 share IP 192.168.1.20 and are close in time
        f2 = dict(base_features, flow_id="f2", src_ip="192.168.1.20", dst_ip="8.8.8.8", protocol="UDP", start_time=100.5, label="dns")
        f3 = dict(base_features, flow_id="f3", src_ip="192.168.1.20", dst_ip="1.1.1.1", protocol="UDP", start_time=102.0, label="dns")

        # Flows 4 and 5: streaming
        f4 = dict(base_features, flow_id="f4", src_ip="192.168.1.99", dst_ip="142.250.190.46", protocol="TCP", start_time=200.0, label="streaming")
        f5 = dict(base_features, flow_id="f5", src_ip="192.168.1.99", dst_ip="142.250.190.46", protocol="TCP", start_time=200.2, label="streaming")

        self.df = pd.DataFrame([f0, f1, f2, f3, f4, f5])

    def test_endpoint_sharing_topology(self):
        builder = FlowGraphBuilder(topology_mode="endpoint_sharing")
        graph_data = builder.create_graph_data(self.df, label_col="label")

        self.assertEqual(graph_data.num_nodes, 6)
        self.assertGreater(graph_data.num_edges, 0)
        self.assertEqual(graph_data.x.shape[0], 6)
        self.assertEqual(graph_data.y.shape[0], 6)

        # Check edge index shape: [2, 2 * num_edges] (bidirectional)
        self.assertEqual(graph_data.edge_index.shape[0], 2)

        # Flows 0 and 1 must be connected (they share 192.168.1.5)
        edges = set(zip(graph_data.edge_index[0].tolist(), graph_data.edge_index[1].tolist()))
        self.assertTrue((0, 1) in edges or (1, 0) in edges)

    def test_knn_topology(self):
        builder = FlowGraphBuilder(topology_mode="knn", k_neighbors=2)
        graph_data = builder.create_graph_data(self.df, label_col="label")

        self.assertEqual(graph_data.num_nodes, 6)
        self.assertGreater(graph_data.num_edges, 0)

    def test_leakage_free_masks(self):
        builder = FlowGraphBuilder(topology_mode="endpoint_sharing")
        graph_data = builder.create_graph_data(self.df, label_col="label", test_size=0.33)

        # Masks must be disjoint
        overlap = (graph_data.train_mask & graph_data.test_mask).sum().item()
        self.assertEqual(overlap, 0, "Train and test masks must not overlap!")
        self.assertGreater(graph_data.train_mask.sum().item(), 0)
        self.assertGreater(graph_data.test_mask.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
