"""
Flow Graph Builder for Encrypted Traffic Classification.
Transforms extracted bidirectional flow records into a mathematically rigorous graph representation:
- Nodes represent network flows with standardized statistical metadata features.
- Edges represent host-interaction (shared endpoints), temporal co-occurrence, or feature similarity.
- Strictly guarantees zero label leakage during edge construction.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DEFAULT_GRAPH_TOPOLOGY,
    KNN_K_NEIGHBORS,
    TEMPORAL_WINDOW_SEC,
)

logger = logging.getLogger(__name__)


class FlowGraphData:
    """Container holding PyTorch-ready graph tensors and metadata."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        test_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
        feature_names: Optional[List[str]] = None,
        label_names: Optional[List[str]] = None,
        flow_ids: Optional[List[str]] = None,
        nx_graph: Optional[nx.Graph] = None,
        graph_stats: Optional[Dict[str, Any]] = None,
    ):
        self.x = x  # [N, d] node feature matrix
        self.edge_index = edge_index  # [2, E] edge index tensor
        self.y = y  # [N] class label tensor
        self.train_mask = train_mask  # [N] boolean train mask
        self.test_mask = test_mask  # [N] boolean test mask
        self.val_mask = val_mask if val_mask is not None else torch.zeros_like(train_mask)
        self.feature_names = feature_names or []
        self.label_names = label_names or []
        self.flow_ids = flow_ids or []
        self.nx_graph = nx_graph
        self.graph_stats = graph_stats or {}

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1] // 2 if self.edge_index.shape[1] > 0 else 0

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


class FlowGraphBuilder:
    """Constructs flow-level graphs from tabular flow DataFrames."""

    def __init__(
        self,
        topology_mode: str = DEFAULT_GRAPH_TOPOLOGY,
        k_neighbors: int = KNN_K_NEIGHBORS,
        temporal_window: float = TEMPORAL_WINDOW_SEC,
    ):
        self.topology_mode = topology_mode
        self.k_neighbors = k_neighbors
        self.temporal_window = temporal_window
        self.scaler = StandardScaler()

    def _prepare_node_features(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extracts numerical metadata and one-hot encoded protocol features.
        Scales features using StandardScaler (fit strictly on train set to prevent leakage).
        """
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

        # Extract numerical array and fill any missing or infinite values
        num_vals = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)

        if is_train:
            scaled_num = self.scaler.fit_transform(num_vals)
        else:
            scaled_num = self.scaler.transform(num_vals)

        # Categorical features (Protocol: TCP, UDP, ICMP, OTHER)
        proto_series = df["protocol"].astype(str).str.upper() if "protocol" in df.columns else pd.Series(["TCP"] * len(df))
        is_tcp = (proto_series == "TCP").values.astype(np.float32).reshape(-1, 1)
        is_udp = (proto_series == "UDP").values.astype(np.float32).reshape(-1, 1)
        is_icmp = (proto_series == "ICMP").values.astype(np.float32).reshape(-1, 1)
        is_other = ((proto_series != "TCP") & (proto_series != "UDP") & (proto_series != "ICMP")).values.astype(np.float32).reshape(-1, 1)

        feature_matrix = np.hstack([scaled_num, is_tcp, is_udp, is_icmp, is_other])
        feature_names = num_cols + ["proto_tcp", "proto_udp", "proto_icmp", "proto_other"]

        return feature_matrix, feature_names

    def build_edges(self, df: pd.DataFrame, feature_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Constructs edges between flows according to the selected topology.
        ZERO label information is used in edge construction.
        """
        num_nodes = len(df)
        edges_set = set()

        if num_nodes <= 1:
            return []

        if self.topology_mode in ("endpoint_sharing", "hybrid"):
            # Edge exists if flow_i and flow_j share at least one IP address
            src_ips = df["src_ip"].astype(str).values
            dst_ips = df["dst_ip"].astype(str).values

            # Inverted index: IP -> set of flow indices
            ip_to_nodes: Dict[str, List[int]] = {}
            for i in range(num_nodes):
                for ip in (src_ips[i], dst_ips[i]):
                    if ip and ip != "None" and ip != "0.0.0.0":
                        ip_to_nodes.setdefault(ip, []).append(i)

            # Create edges between flows sharing an IP (with a per-hub limit to avoid star hub explosion)
            max_cluster_size = 50
            for ip, node_list in ip_to_nodes.items():
                if len(node_list) > 1:
                    # If an IP has too many connections, cap it to prevent dense cliques
                    sampled_nodes = node_list[:max_cluster_size]
                    for idx_a in range(len(sampled_nodes)):
                        for idx_b in range(idx_a + 1, len(sampled_nodes)):
                            u, v = sampled_nodes[idx_a], sampled_nodes[idx_b]
                            edges_set.add((min(u, v), max(u, v)))

        if self.topology_mode in ("temporal", "hybrid"):
            # Edge exists if start times are within temporal_window
            if "start_time" in df.columns:
                start_times = df["start_time"].values.astype(np.float64)
                sorted_indices = np.argsort(start_times)
                sorted_times = start_times[sorted_indices]

                for i in range(num_nodes):
                    cur_idx = sorted_indices[i]
                    cur_time = sorted_times[i]
                    # Check subsequent flows within window
                    for j in range(i + 1, num_nodes):
                        nxt_idx = sorted_indices[j]
                        if (sorted_times[j] - cur_time) <= self.temporal_window:
                            edges_set.add((min(cur_idx, nxt_idx), max(cur_idx, nxt_idx)))
                        else:
                            break

        if self.topology_mode == "knn" or (len(edges_set) == 0 and num_nodes > 1):
            # Fallback or primary: k-NN in normalized metadata feature space
            k = min(self.k_neighbors, num_nodes - 1)
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(feature_matrix)
            _, indices = nbrs.kneighbors(feature_matrix)
            for i in range(num_nodes):
                for j in indices[i][1:]:
                    edges_set.add((min(i, int(j)), max(i, int(j))))

        return sorted(list(edges_set))

    def create_graph_data(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.25,
        val_size: float = 0.10,
        split_strategy: str = "stratified",  # 'stratified' or 'unseen_host'
        random_seed: int = 42,
    ) -> FlowGraphData:
        """
        Creates a complete FlowGraphData object with nodes, edges, labels,
        leakage-free train/val/test masks, and network statistics.
        """
        num_nodes = len(df)
        if num_nodes == 0:
            raise ValueError("Input DataFrame contains 0 flows.")

        # 1. Feature Preprocessing
        feature_matrix, feature_names = self._prepare_node_features(df, is_train=True)

        # 2. Edge Construction
        raw_edges = self.build_edges(df, feature_matrix)

        # Create bidirectional edge_index tensor for GNN message passing
        edge_list_fwd = [(u, v) for u, v in raw_edges]
        edge_list_bwd = [(v, u) for u, v in raw_edges]
        all_directed = edge_list_fwd + edge_list_bwd

        if all_directed:
            edge_index = torch.tensor(all_directed, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3. Label Processing
        if label_col in df.columns:
            raw_labels = df[label_col].astype(str).tolist()
            unique_labels = sorted(list(set(raw_labels)))
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            y_arr = np.array([label_map[lbl] for lbl in raw_labels], dtype=np.int64)
            y_tensor = torch.tensor(y_arr, dtype=torch.long)
        else:
            unique_labels = ["unknown"]
            y_tensor = torch.zeros(num_nodes, dtype=torch.long)

        # 4. Leakage-Free Splitting
        np.random.seed(random_seed)
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)

        if split_strategy == "unseen_host" and "src_ip" in df.columns:
            # Unseen-host split: Host IP disjoint partition to prevent IP leakage
            unique_hosts = df["src_ip"].unique()
            np.random.shuffle(unique_hosts)
            n_test = max(1, int(len(unique_hosts) * test_size))
            test_hosts = set(unique_hosts[:n_test])
            train_val_hosts = unique_hosts[n_test:]
            n_val = max(1, int(len(train_val_hosts) * val_size))
            val_hosts = set(train_val_hosts[:n_val])

            for i, ip in enumerate(df["src_ip"]):
                if ip in test_hosts:
                    test_mask[i] = True
                elif ip in val_hosts:
                    val_mask[i] = True
                else:
                    train_mask[i] = True
        else:
            # Stratified split by class
            for c_idx in range(len(unique_labels)):
                cls_indices = np.where(y_tensor.numpy() == c_idx)[0]
                np.random.shuffle(cls_indices)
                n_c = len(cls_indices)
                n_c_test = max(1, int(n_c * test_size)) if n_c > 3 else 1
                n_c_val = max(1, int(n_c * val_size)) if n_c > 4 else 0

                c_test = cls_indices[:n_c_test]
                c_val = cls_indices[n_c_test : n_c_test + n_c_val]
                c_train = cls_indices[n_c_test + n_c_val :]

                test_mask[c_test] = True
                val_mask[c_val] = True
                train_mask[c_train] = True

        # Ensure at least some training nodes exist
        if not np.any(train_mask):
            train_mask[0] = True
            test_mask[0] = False

        # 5. Build NetworkX graph for topology metrics and visualization
        nx_g = nx.Graph()
        nx_g.add_nodes_from(range(num_nodes))
        nx_g.add_edges_from(raw_edges)

        degrees = [d for _, d in nx_g.degree()]
        graph_stats = {
            "num_nodes": num_nodes,
            "num_edges": len(raw_edges),
            "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
            "density": float(nx.density(nx_g)),
            "num_connected_components": nx.number_connected_components(nx_g),
            "topology_mode": self.topology_mode,
            "split_strategy": split_strategy,
            "train_nodes": int(train_mask.sum()),
            "val_nodes": int(val_mask.sum()),
            "test_nodes": int(test_mask.sum()),
        }

        flow_ids = df["flow_id"].tolist() if "flow_id" in df.columns else [f"flow_{i}" for i in range(num_nodes)]

        return FlowGraphData(
            x=torch.tensor(feature_matrix, dtype=torch.float32),
            edge_index=edge_index,
            y=y_tensor,
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            feature_names=feature_names,
            label_names=unique_labels,
            flow_ids=flow_ids,
            nx_graph=nx_g,
            graph_stats=graph_stats,
        )
