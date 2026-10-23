"""
GraphSAGE: Inductive Representation Learning on Large Graphs (Hamilton et al., NeurIPS 2017).
Mathematically authentic PyTorch implementation of GraphSAGE:
- Neighborhood Mean Aggregator
- Concatenation of target node embedding and neighborhood representation
- Linear weight transformation with non-linear activation (ReLU)
- L2-normalization of output embeddings
- Multi-layer architecture with inductive inference support
"""
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GRAPHSAGE_CONFIG

logger = logging.getLogger(__name__)


class SAGEConv(nn.Module):
    """
    GraphSAGE Convolutional Layer with Mean Aggregator (Hamilton et al., 2017, Eq. 1-3).
    h_{N(v)} = Mean_{u in N(v)}(h_u)
    h_v' = sigma(W * [h_v || h_{N(v)}] + b)
    h_v' = h_v' / ||h_v'||_2
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Linear transform over concatenated [self_features || neighbor_features]
        self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_features]
        edge_index: [2, E] (row 0: source nodes u, row 1: target nodes v)
        """
        num_nodes = x.size(0)

        # 1. Neighborhood Mean Aggregation
        # Sum neighbor features into each target node
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            # Accumulate neighbor messages
            msg = x[src]  # [E, in_features]
            neigh_sum = torch.zeros_like(x)
            neigh_sum.index_add_(0, dst, msg)

            # Compute in-degree for each node
            deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
            ones = torch.ones(src.size(0), dtype=x.dtype, device=x.device)
            deg.index_add_(0, dst, ones)
            deg = torch.clamp(deg, min=1.0).unsqueeze(1)  # avoid div by zero

            neigh_mean = neigh_sum / deg
        else:
            # Isolated graph / no edges: neighbor representation is zero
            neigh_mean = torch.zeros_like(x)

        # 2. Concatenate target node representation with neighborhood representation
        # [h_v || h_{N(v)}]
        concat_feat = torch.cat([x, neigh_mean], dim=1)  # [N, 2 * in_features]

        # 3. Linear projection + non-linear activation
        out = self.linear(concat_feat)

        # 4. L2 Normalization (Hamilton et al. 2017)
        out = F.normalize(out, p=2, dim=1)
        return out


class GraphSAGEClassifier(nn.Module):
    """
    2-Layer GraphSAGE Network followed by an MLP Readout Head for flow classification.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        # Readout classification head
        logits = self.classifier(h)
        return logits

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns the final GNN node embeddings before the classification head."""
        with torch.no_grad():
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
        return h


class GraphSAGETrainer:
    """Manages training, validation, early stopping, and evaluation for GraphSAGE."""

    def __init__(self, model: GraphSAGEClassifier, config: Optional[Dict[str, Any]] = None):
        self.config = config or GRAPHSAGE_CONFIG
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Executes inductive / transductive masked training loop with early stopping.
        """
        epochs = epochs or self.config["epochs"]
        patience = patience or self.config["patience"]

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        y = y.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        has_val = val_mask.sum().item() > 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(x, edge_index)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            self.optimizer.step()

            # Train accuracy
            preds = logits.argmax(dim=-1)
            train_acc = (preds[train_mask] == y[train_mask]).float().mean().item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x, edge_index)
                if has_val:
                    val_loss = self.criterion(val_logits[val_mask], y[val_mask]).item()
                    val_preds = val_logits.argmax(dim=-1)
                    val_acc = (val_preds[val_mask] == y[val_mask]).float().mean().item()
                else:
                    val_loss = loss.item()
                    val_acc = train_acc

            history["train_loss"].append(loss.item())
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= 20:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Returns predicted class labels and probability distributions."""
        self.model.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
        return preds, probs
