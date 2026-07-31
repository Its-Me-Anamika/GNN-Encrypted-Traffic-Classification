import numpy as np
import torch

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class GraphBuilder:
    """
    Constructs Graph Neural Network Data Objects from a collection of network flows.
    Nodes = Flow 5-tuples (src_ip, src_port, dst_ip, dst_port, protocol)
    Edges = Shared IP relationships (src_ip or dst_ip match) with temporal proximity weight
    """

    @staticmethod
    def build_flow_graph(flows_list):
        """
        Builds a PyTorch Geometric graph data object from a list of flow dictionaries or DB models.
        """
        num_nodes = len(flows_list)
        if num_nodes == 0:
            x = torch.zeros((0, 28), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr) if HAS_PYG else {"x": x, "edge_index": edge_index}

        node_features = []
        for flow in flows_list:
            if hasattr(flow, "feature_vector") and flow.feature_vector is not None:
                feat = flow.feature_vector
            elif isinstance(flow, dict) and "features" in flow:
                feat = flow["features"]
            else:
                from app.utils.preprocessing import extract_statistical_features
                feat = extract_statistical_features()
            node_features.append(feat)

        x = torch.tensor(np.array(node_features, dtype=np.float32), dtype=torch.float32)

        # Construct Edge Index based on 5-tuple IP relationships & temporal proximity
        edges = []
        edge_weights = []

        for i in range(num_nodes):
            flow_i = flows_list[i]
            src_i = getattr(flow_i, "src_ip", None) or flow_i.get("src_ip", "")
            dst_i = getattr(flow_i, "dst_ip", None) or flow_i.get("dst_ip", "")
            dur_i = getattr(flow_i, "duration", None) or flow_i.get("duration", 1.0)

            for j in range(i + 1, num_nodes):
                flow_j = flows_list[j]
                src_j = getattr(flow_j, "src_ip", None) or flow_j.get("src_ip", "")
                dst_j = getattr(flow_j, "dst_ip", None) or flow_j.get("dst_ip", "")
                dur_j = getattr(flow_j, "duration", None) or flow_j.get("duration", 1.0)

                # Edge condition: Shared source IP, destination IP, or cross-talking IP
                if (src_i == src_j) or (dst_i == dst_j) or (src_i == dst_j) or (dst_i == src_j):
                    edges.append([i, j])
                    edges.append([j, i])

                    # Temporal proximity weight w = exp(-|dur_i - dur_j| / tau)
                    time_diff = abs(dur_i - dur_j)
                    w = float(np.exp(-time_diff / 10.0))
                    edge_weights.append([w])
                    edge_weights.append([w])

        if not edges:
            # Self loops if graph has no IP overlap
            for i in range(num_nodes):
                edges.append([i, i])
                edge_weights.append([1.0])

        edge_index = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_weights, dtype=np.float32), dtype=torch.float32)

        if HAS_PYG:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        else:
            return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
