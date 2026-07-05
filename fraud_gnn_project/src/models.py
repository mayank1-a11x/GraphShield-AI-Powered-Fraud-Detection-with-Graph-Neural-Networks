"""GNN model definitions for fraud detection.

Three architectures, matching the project's build order:
  - GATFraudNet: homogeneous graph, attention over transaction-transaction
    edges. Fast baseline, easy to interpret attention weights.
  - RGCNFraudNet: heterogeneous graph, one weight matrix per relation type.
  - HGTFraudNet: heterogeneous graph, learned attention per node/edge type
    (heavier, usually the strongest of the three).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, RGCNConv, HGTConv, Linear as PyGLinear


class FocalLoss(nn.Module):
    """Focal loss down-weights easy (well-classified) examples so the model
    keeps learning from the rare, hard fraud cases instead of being swamped
    by the ~96% non-fraud majority."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


class GATFraudNet(nn.Module):
    """GAT over the homogeneous transaction-transaction projected graph."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=num_heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * num_heads, hidden_dim,
                                         heads=num_heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        self.convs.append(GATv2Conv(hidden_dim * num_heads, hidden_dim,
                                     heads=1, dropout=dropout, concat=False))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # residual only when shapes match (first layer usually won't)
            if x.shape == x_res.shape:
                x = x + x_res
        return self.classifier(x)


class RGCNFraudNet(nn.Module):
    """RGCN over the heterogeneous graph. Expects the graph to be converted
    to a single relation-typed edge_index (via HeteroData.to_homogeneous()
    or an equivalent mapping done in train.py) with an edge_type tensor."""

    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class HGTFraudNet(nn.Module):
    """Heterogeneous Graph Transformer. Operates directly on HeteroData-style
    dicts of node features / edge indices, with per-type attention."""

    def __init__(self, node_types: list[str], edge_types: list[tuple],
                 in_dims: dict[str, int], hidden_dim: int = 128,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Project every node type's raw features into a shared hidden dim.
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            in_dim = in_dims.get(node_type, 1)
            self.lin_dict[node_type] = PyGLinear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_dim, hidden_dim, (node_types, edge_types),
                        heads=num_heads)
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h_dict = {nt: self.lin_dict[nt](x).relu() for nt, x in x_dict.items()}
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {nt: F.dropout(h, p=self.dropout, training=self.training)
                      for nt, h in h_dict.items()}
        return self.classifier(h_dict["transaction"])


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Factory so train.py stays agnostic to which architecture is active."""
    if model_type == "gat":
        return GATFraudNet(**kwargs)
    if model_type == "rgcn":
        return RGCNFraudNet(**kwargs)
    if model_type == "hgt":
        return HGTFraudNet(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")
