"""
Molecular Encoder using E(3)-equivariant Graph Neural Networks.

Implements compositional pooling to separate:
- Scaffold embeddings (graph motifs)
- Functional group embeddings (learned chemical grammar)
- 3D geometric features (when available)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from torch_geometric.nn.pool import global_add_pool as gap
from torch_geometric.nn.pool import global_max_pool as gmp
from e3nn import o3
from e3nn.nn import Gate
from typing import Optional, Tuple


def scatter_mps_compatible(src, index, dim=0, dim_size=None, reduce='sum'):
    """
    MPS-compatible scatter operation using native PyTorch.

    Replaces torch_scatter.scatter() which doesn't support Apple Silicon MPS.
    Uses PyTorch's native operations that work on CPU, CUDA, and MPS.

    Args:
        src: Source tensor to scatter [N, ...]
        index: Index tensor [N] indicating where to scatter each element
        dim: Dimension along which to scatter (default: 0)
        dim_size: Size of output dimension (default: max(index) + 1)
        reduce: Reduction operation ('sum', 'mean', or 'max')

    Returns:
        Scattered tensor [dim_size, ...]
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # Prepare output shape
    shape = list(src.shape)
    shape[dim] = dim_size

    if reduce == 'sum':
        # Simple scatter-add operation
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        # Expand index to match src dimensions
        index_expanded = index
        for _ in range(len(src.shape) - 1):
            index_expanded = index_expanded.unsqueeze(-1)
        index_expanded = index_expanded.expand_as(src)
        return out.scatter_add_(dim, index_expanded, src)

    elif reduce == 'mean':
        # Scatter-add then divide by count
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        count = torch.zeros(dim_size, dtype=torch.long, device=src.device)

        # Scatter values
        index_expanded = index
        for _ in range(len(src.shape) - 1):
            index_expanded = index_expanded.unsqueeze(-1)
        index_expanded = index_expanded.expand_as(src)
        out.scatter_add_(dim, index_expanded, src)

        # Count occurrences
        count.scatter_add_(0, index, torch.ones_like(index, dtype=torch.long))

        # Divide by count (with broadcasting)
        count_expanded = count
        for _ in range(len(src.shape) - 1):
            count_expanded = count_expanded.unsqueeze(-1)

        return out / count_expanded.clamp(min=1)

    elif reduce == 'max':
        # Use native scatter_reduce for max (PyTorch 2.0+)
        # Initialize with very negative values for max operation
        out = torch.full(shape, float('-inf'), dtype=src.dtype, device=src.device)

        index_expanded = index
        for _ in range(len(src.shape) - 1):
            index_expanded = index_expanded.unsqueeze(-1)
        index_expanded = index_expanded.expand_as(src)

        return out.scatter_reduce_(dim, index_expanded, src, reduce='amax', include_self=False)

    else:
        raise ValueError(f"Unsupported reduce operation: {reduce}. Use 'sum', 'mean', or 'max'.")


class E3EquivariantConv(MessagePassing):
    """
    E(3)-equivariant message passing layer.
    Preserves rotational and translational symmetry for 3D molecular structures.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int = 0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Position update (equivariant)
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, in_dim]
            pos: Node positions [N, 3]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]

        Returns:
            Updated node features and positions
        """
        # Message passing
        out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        # Update features
        x_new = self.update_net(torch.cat([x, out], dim=-1))

        # Update positions (equivariant)
        row, col = edge_index
        pos_diff = pos[row] - pos[col]
        pos_diff_norm = pos_diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        pos_dir = pos_diff / pos_diff_norm

        # Aggregate position updates with additional safety checks
        pos_update = self.pos_net(out).unsqueeze(-1) * pos_dir
        # Clamp position updates to prevent explosion
        pos_update = torch.clamp(pos_update, min=-10.0, max=10.0)
        pos_new = pos + scatter_mps_compatible(pos_update, col, dim=0, dim_size=pos.size(0), reduce='mean')

        return x_new, pos_new

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Compute distance
        pos_diff = pos_i - pos_j
        dist = pos_diff.norm(dim=-1, keepdim=True)
        # Clamp distance to prevent zero division and extreme values
        dist = torch.clamp(dist, min=1e-4, max=100.0)

        # Build message
        msg_input = [x_i, x_j, dist]
        if edge_attr is not None:
            msg_input.append(edge_attr)

        msg = torch.cat(msg_input, dim=-1)
        return self.message_net(msg)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for local chemistry embeddings.
    Learns to weight different atoms by importance.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, dim]
            batch: Batch assignment [N]

        Returns:
            Pooled features [B, dim]
        """
        B = batch.max().item() + 1

        # Project to keys and values
        k = self.key(x).view(-1, self.num_heads, self.head_dim)  # [N, H, D]
        v = self.value(x).view(-1, self.num_heads, self.head_dim)  # [N, H, D]

        # Compute attention scores
        q = self.query.unsqueeze(0)  # [1, H, D]
        attn = (q * k).sum(dim=-1) * self.scale  # [N, H]

        # Clamp attention logits to prevent overflow (NUMERICAL STABILITY FIX)
        attn = torch.clamp(attn, min=-20.0, max=20.0)

        # Softmax per graph (using log-sum-exp trick for stability)
        attn_max = scatter_mps_compatible(attn, batch, dim=0, dim_size=B, reduce='max')[batch]  # [N, H]
        attn_shifted = attn - attn_max
        attn_exp = torch.exp(attn_shifted)
        attn_sum = scatter_mps_compatible(attn_exp, batch, dim=0, dim_size=B, reduce='sum')[batch]  # [N, H]
        attn_norm = attn_exp / (attn_sum + 1e-8)

        # Weighted sum
        weighted = attn_norm.unsqueeze(-1) * v  # [N, H, D]
        pooled = scatter_mps_compatible(weighted, batch, dim=0, dim_size=B, reduce='sum')  # [B, H, D]

        return pooled.view(B, -1)


class Set2Set(nn.Module):
    """
    Set2Set pooling for global topology embeddings.
    Processes the entire molecular graph as a set.
    """

    def __init__(self, in_dim: int, processing_steps: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 2 * in_dim
        self.processing_steps = processing_steps

        self.lstm = nn.LSTM(self.out_dim, in_dim, batch_first=True)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            batch: Batch assignment [N]

        Returns:
            Pooled features [B, 2*in_dim]
        """
        B = batch.max().item() + 1

        # Initialize query
        h = (
            x.new_zeros((1, B, self.in_dim)),
            x.new_zeros((1, B, self.in_dim)),
        )

        q_star = x.new_zeros(B, self.out_dim)

        for _ in range(self.processing_steps):
            # LSTM step
            q, h = self.lstm(q_star.unsqueeze(1), h)
            q = q.squeeze(1)  # [B, in_dim]

            # Attention
            q_expanded = q[batch]  # [N, in_dim]
            attn = (x * q_expanded).sum(dim=-1)  # [N]

            # Clamp attention logits to prevent overflow (NUMERICAL STABILITY FIX)
            attn = torch.clamp(attn, min=-20.0, max=20.0)

            # Softmax per graph (using log-sum-exp trick for stability)
            attn_max = scatter_mps_compatible(attn, batch, dim=0, dim_size=B, reduce='max')[batch]
            attn_shifted = attn - attn_max
            attn_exp = torch.exp(attn_shifted)
            attn_sum = scatter_mps_compatible(attn_exp, batch, dim=0, dim_size=B, reduce='sum')[batch]
            attn_norm = attn_exp / (attn_sum + 1e-8)

            # Weighted sum
            r = scatter_mps_compatible(attn_norm.unsqueeze(-1) * x, batch, dim=0, dim_size=B, reduce='sum')

            # Update query
            q_star = torch.cat([q, r], dim=-1)

        return q_star


class MolecularEncoder(nn.Module):
    """
    Molecular encoder with compositional pooling.

    Encodes molecules into hierarchical embeddings:
    - z_mol_local: Local chemistry (atom neighborhoods, functional groups)
    - z_mol_global: Global topology (scaffold, overall structure)

    Args:
        atom_feature_dim: Dimension of input atom features
        edge_feature_dim: Dimension of input edge features
        hidden_dim: Hidden dimension for GNN layers
        local_dim: Output dimension for local embeddings (default: 512)
        global_dim: Output dimension for global embeddings (default: 256)
        num_layers: Number of message passing layers (default: 5)
        use_3d: Whether to use 3D coordinates (default: True)
    """

    def __init__(
        self,
        atom_feature_dim: int = 128,
        edge_feature_dim: int = 32,
        hidden_dim: int = 512,
        local_dim: int = 512,
        global_dim: int = 256,
        num_layers: int = 5,
        use_3d: bool = True,
    ):
        super().__init__()

        self.atom_feature_dim = atom_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_layers = num_layers
        self.use_3d = use_3d

        # Input embedding with LayerNorm for better gradient flow
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_embedding = nn.Linear(edge_feature_dim, edge_feature_dim) if edge_feature_dim > 0 else None

        # E(3)-equivariant layers (if using 3D)
        if use_3d:
            self.conv_layers = nn.ModuleList([
                E3EquivariantConv(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=edge_feature_dim,
                )
                for _ in range(num_layers)
            ])
        else:
            # Standard message passing for 2D graphs
            from torch_geometric.nn import GINEConv
            self.conv_layers = nn.ModuleList([
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=edge_feature_dim,
                )
                for _ in range(num_layers)
            ])

        # Pooling layers
        self.local_pool = AttentionPooling(hidden_dim, num_heads=8)
        self.global_pool = Set2Set(hidden_dim, processing_steps=3)

        # Output projections
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, local_dim),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, global_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Atom features [N, atom_feature_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            edge_attr: Edge features [E, edge_feature_dim]
            pos: 3D coordinates [N, 3] (required if use_3d=True)

        Returns:
            z_local: Local chemistry embeddings [B, local_dim]
            z_global: Global topology embeddings [B, global_dim]
        """
        # Embed inputs
        h = self.atom_embedding(x)

        if edge_attr is not None and self.edge_embedding is not None:
            edge_attr = self.edge_embedding(edge_attr)

        # Message passing with safer residual connections (NUMERICAL STABILITY FIX)
        if self.use_3d:
            assert pos is not None, "3D coordinates required when use_3d=True"
            for conv in self.conv_layers:
                h_new, pos = conv(h, pos, edge_index, edge_attr)
                # Clamp updates to prevent explosion
                h_new = torch.clamp(h_new, min=-10.0, max=10.0)
                # Gated residual: reduce contribution of new features
                h = 0.9 * h + 0.1 * h_new
        else:
            for conv in self.conv_layers:
                h_new = conv(h, edge_index, edge_attr)
                # Clamp updates to prevent explosion
                h_new = torch.clamp(h_new, min=-10.0, max=10.0)
                # Gated residual: reduce contribution of new features
                h = 0.9 * h + 0.1 * h_new

        # Compositional pooling
        z_local = self.local_pool(h, batch)
        z_global = self.global_pool(h, batch)

        # Project to output dimensions
        z_local = self.local_proj(z_local)
        z_global = self.global_proj(z_global)

        return z_local, z_global

    @property
    def output_dim(self) -> int:
        """Total output dimension (local + global)."""
        return self.local_dim + self.global_dim
