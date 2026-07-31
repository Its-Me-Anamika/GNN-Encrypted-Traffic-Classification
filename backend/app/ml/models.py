import torch
import torch.nn as nn
import torch.nn.functional as F

# Try importing PyTorch Geometric; provide clean fallback if not installed
try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class FallbackGNNLayer(nn.Module):
    """Fallback Graph Layer using Dense Adjacency for environments without PyG."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / deg
        out = torch.matmul(norm_adj, x)
        return self.linear(out)

class GraphSAGEModel(nn.Module):
    """GraphSAGE Architecture with Skip Connections and BatchNorm."""
    def __init__(self, in_channels=28, hidden_channels=128, out_channels=9, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        if HAS_PYG:
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        else:
            self.conv1 = FallbackGNNLayer(in_channels, hidden_channels)
            self.conv2 = FallbackGNNLayer(hidden_channels, hidden_channels)
            self.conv3 = FallbackGNNLayer(hidden_channels, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        # Layer 2 with Skip Connection
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2 + h1)  # Residual connection
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)

        # Layer 3 with Skip Connection
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.relu(h3 + h2)  # Residual connection

        # Classification Head
        logits = self.classifier(h3)
        return logits

class GATModel(nn.Module):
    """Graph Attention Network (GAT) Architecture with Multi-Head Attention."""
    def __init__(self, in_channels=28, hidden_channels=64, out_channels=9, heads=4, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        if HAS_PYG:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
            self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout_rate)
        else:
            self.conv1 = FallbackGNNLayer(in_channels, hidden_channels * heads)
            self.conv2 = FallbackGNNLayer(hidden_channels * heads, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, batch=None):
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.elu(h2)

        logits = self.classifier(h2)
        return logits

class EncryptedTrafficGNN(nn.Module):
    """
    Unified GNN Wrapper supporting Monte Carlo Dropout for Uncertainty Estimation
    and baseline switching between GraphSAGE and GAT.
    """
    def __init__(self, architecture="GraphSAGE", in_channels=28, out_channels=9, dropout_rate=0.3):
        super().__init__()
        self.architecture = architecture
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        if architecture.lower() == "gat":
            self.gnn = GATModel(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate)
        else:
            self.gnn = GraphSAGEModel(in_channels=in_channels, out_channels=out_channels, dropout_rate=dropout_rate)

    def forward(self, x, edge_index, batch=None):
        return self.gnn(x, edge_index, batch)

    def predict_with_mc_dropout(self, x, edge_index, num_samples=20):
        """
        Monte Carlo Dropout inference while keeping BatchNorm layers in eval mode.
        """

        # Put entire model in eval mode
        self.eval()

        # Enable ONLY dropout layers
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        probs_list = []

        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(x, edge_index)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.unsqueeze(0))

        stacked_probs = torch.cat(probs_list, dim=0)

        mean_probs = torch.mean(stacked_probs, dim=0)
        variance = torch.var(stacked_probs, dim=0)

        uncertainty_scores = torch.mean(variance, dim=-1)
        confidences, predictions = torch.max(mean_probs, dim=-1)

        return predictions, confidences, uncertainty_scores, mean_probs
