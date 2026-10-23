"""
Graph Attention Network (GAT) for Encrypted Traffic Flow Classification (Veličković et al., ICLR 2018).
Investigates self-attention over flow graph edges to dynamically weigh important neighbor conversations.
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import GAT_CONFIG

logger = logging.getLogger(__name__)


class GATConv(nn.Module):
    """
    Multi-Head Graph Attention Layer (Veličković et al., 2018).
    Computes masked self-attention over flow graph neighbors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        alpha: float = 0.2,
        concat: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat

        # Linear projection for each head: [in_features, num_heads * out_features]
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)

        # Attention vectors for source and target nodes
        self.a_src = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_dst = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)

        # Linear transform: [N, num_heads, out_features]
        h = self.W(x).view(num_nodes, self.num_heads, self.out_features)

        # Add self-loops if needed to allow attention on self
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
        else:
            # Empty graph: fall back to self-linear projection
            if self.concat:
                return h.view(num_nodes, self.num_heads * self.out_features)
            else:
                return h.mean(dim=1)

        # Self-attention scoring
        # score = LeakyReLU(a_src * h_src + a_dst * h_dst)
        score_src = (h * self.a_src).sum(dim=-1)  # [N, num_heads]
        score_dst = (h * self.a_dst).sum(dim=-1)  # [N, num_heads]

        edge_scores = score_src[src] + score_dst[dst]  # [E, num_heads]
        edge_scores = self.leaky_relu(edge_scores)

        # Softmax over incoming edges for each destination node
        # Subtract max for numerical stability
        exp_scores = torch.exp(edge_scores - edge_scores.max(dim=0, keepdim=True)[0])
        denom = torch.zeros(num_nodes, self.num_heads, device=x.device)
        denom.index_add_(0, dst, exp_scores)
        denom = torch.clamp(denom, min=1e-8)

        attn_weights = exp_scores / denom[dst]  # [E, num_heads]
        attn_weights = self.dropout(attn_weights)

        # Message aggregation: weighted sum of source embeddings
        # [E, num_heads, out_features]
        weighted_msgs = h[src] * attn_weights.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        out.index_add_(0, dst, weighted_msgs)

        if self.concat:
            return out.view(num_nodes, self.num_heads * self.out_features)
        else:
            return out.mean(dim=1)


class GATClassifier(nn.Module):
    """2-Layer Multi-Head Graph Attention Network for Flow Classification."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        embed_dim: int = 32,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Layer 1: multi-head attention concatenated
        self.gat1 = GATConv(in_features, hidden_dim // num_heads, num_heads=num_heads, concat=True, dropout=dropout)
        # Layer 2: multi-head attention averaged
        self.gat2 = GATConv(hidden_dim, embed_dim, num_heads=num_heads, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        logits = self.classifier(h)
        return logits


class GATTrainer:
    """Trainer for Graph Attention Networks."""

    def __init__(self, model: GATClassifier, config: Optional[Dict[str, Any]] = None):
        self.config = config or GAT_CONFIG
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

            preds = logits.argmax(dim=-1)
            train_acc = (preds[train_mask] == y[train_mask]).float().mean().item()

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= 20:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
        return preds, probs
