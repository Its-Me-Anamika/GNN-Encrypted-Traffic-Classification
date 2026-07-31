import torch
import torch.nn.functional as F
import numpy as np

from app.utils.preprocessing import FEATURE_NAMES
from app.config import settings

class GNNExplainabilityService:
    """
    Explains GNN predictions by computing feature attribution masks and edge attributions
    for individual traffic flows.
    """

    def __init__(self, model):
        self.model = model

    def explain_flow(self, x, edge_index, target_node_idx=0, top_k_features=8):
        """
        Computes feature attribution importance for a target flow node.
        Uses gradient-based saliency / feature perturbation attribution.
        """
        self.model.eval()

        # Clone input for gradient tracking
        x_input = x.clone().detach().requires_grad_(True)
        logits = self.model(x_input, edge_index)
        
        if target_node_idx >= logits.size(0):
            target_node_idx = 0

        predicted_class = torch.argmax(logits[target_node_idx]).item()
        target_score = logits[target_node_idx, predicted_class]

        # Backward pass to compute feature saliency
        self.model.zero_grad()
        target_score.backward()

        saliency = x_input.grad[target_node_idx].abs().cpu().numpy()

        # Normalize feature importance scores to sum to 100%
        sum_sal = float(np.sum(saliency))
        if sum_sal > 0:
            normalized_saliency = (saliency / sum_sal) * 100.0
        else:
            normalized_saliency = np.full(len(saliency), 100.0 / len(saliency))

        # Build feature attribution list
        feature_importance = []
        for i, fname in enumerate(FEATURE_NAMES):
            if i < len(normalized_saliency):
                feature_importance.append({
                    "feature_name": fname,
                    "importance_score": round(float(normalized_saliency[i]), 2),
                    "raw_value": round(float(x[target_node_idx, i].item()), 4)
                })

        # Sort by importance
        feature_importance.sort(key=lambda item: item["importance_score"], reverse=True)

        # Get connected neighbor sub-graph
        neighbors = []
        if edge_index.size(1) > 0:
            mask = (edge_index[0] == target_node_idx) | (edge_index[1] == target_node_idx)
            connected_edges = edge_index[:, mask]

            for col in range(connected_edges.size(1)):
                u, v = connected_edges[0, col].item(), connected_edges[1, col].item()
                neighbor_idx = v if u == target_node_idx else u
                if neighbor_idx != target_node_idx and neighbor_idx not in neighbors:
                    neighbors.append(int(neighbor_idx))

        return {
            "target_node_index": int(target_node_idx),
            "predicted_category": settings.CLASSES[predicted_class] if predicted_class < len(settings.CLASSES) else "Unknown",
            "top_features": feature_importance[:top_k_features],
            "all_features": feature_importance,
            "connected_neighbors_count": len(neighbors),
            "neighbor_node_indices": neighbors[:10],
            "explanation_summary": f"Classification for flow node #{target_node_idx} is heavily driven by '{feature_importance[0]['feature_name']}' ({feature_importance[0]['importance_score']}%) and '{feature_importance[1]['feature_name']}' ({feature_importance[1]['importance_score']}%)."
        }
