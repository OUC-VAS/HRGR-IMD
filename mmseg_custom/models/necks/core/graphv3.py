from typing import List, Tuple, Dict, TypeVar

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import nn
from torch.nn import Linear

from .graph import CrossAttention


class DenseSAGEConvV3(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.SAGEConv`.

    .. note::

        :class:`~torch_geometric.nn.dense.DenseSAGEConv` expects to work on
        binary adjacency matrices.
        If you want to make use of weighted dense adjacency matrices, please
        use :class:`torch_geometric.nn.dense.DenseGraphConv` instead.

    """

    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_rel = Linear(in_channels, out_channels, bias=False)
        # self.lin_root = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        # self.lin_root.reset_parameters()

    def forward(self, x, adj, x_mask=None, adj_mask=None):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            x_mask (BoolTensor, optional): Node Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
           adj_mask (BoolTensor, optional): Adj Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N \times N}` indicating
                the valid edges for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if x_mask is not None:
            x = x * x_mask.view(B, N, -1).to(x.dtype)

        if adj_mask is not None:
            adj = adj * adj_mask.view(B, N, N).to(x.dtype)

        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        # out = self.lin_rel(out) + self.lin_root(x)
        out = self.lin_rel(out)

        # if x_mask is not None:
        #     out = out * x_mask.view(B, N, 1).to(x.dtype)

        if x_mask is not None:
            out = out * x_mask.view(B, N, -1).to(x.dtype)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


class HeterogeneousAggregatorV3(BaseModule):

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intra_layer = nn.ModuleDict()
        self.inter_layer = nn.ModuleDict()
        self.n_layer = n_layers
        for i in range(self.n_layer):
            self.intra_layer[f'{i}'] = DenseSAGEConvV3(in_channels, out_channels, normalize=True)

        for i in range(self.n_layer - 1):
            self.inter_layer[f'{i}_{i + 1}'] = DenseSAGEConvV3(in_channels, out_channels, normalize=True)
            self.inter_layer[f'{i + 1}_{i}'] = DenseSAGEConvV3(in_channels, out_channels, normalize=True)

        self.layer_aggr = nn.ModuleDict()
        for i in range(self.n_layer):
            self.layer_aggr[f'{i}'] = CrossAttention(out_channels, q_dim=in_channels)

        self.lin_root = Linear(in_channels, out_channels)

    def forward(self, x, adj, slice_index: List[int]):
        """
        Args:
            x: (B, N, C)
            adj: (B, N, N)
            slice_index:

        Returns: (B, N, C)
        """
        B, N, _ = x.shape
        y = torch.zeros((B, N, self.out_channels), dtype=x.dtype, device=x.device)
        for i in range(self.n_layer):

            feat_list = []
            if i > 0:
                l, m, r = slice_index[i - 1], slice_index[i], slice_index[i + 1]
                x_mask = torch.zeros((B, N, 1), dtype=x.dtype, device=x.device)
                x_mask[:, l: r, :] = 1.0
                adj_mask = torch.zeros_like(adj)
                adj_mask[:, m: r, l: m] = 1.0
                lower = self.inter_layer[f'{i - 1}_{i}'](x, adj, x_mask=x_mask, adj_mask=adj_mask)
                feat_list.append(lower)

            l, r = slice_index[i], slice_index[i + 1]
            x_mask = torch.zeros((B, N, 1), dtype=x.dtype, device=x.device)
            x_mask[:, l: r, :] = 1.0
            adj_mask = torch.zeros_like(adj)
            adj_mask[:, l: r, l: r] = 1.0
            curr = self.intra_layer[f'{i}'](x, adj, x_mask=x_mask, adj_mask=adj_mask)
            feat_list.append(curr)

            if i < self.n_layer - 1:
                l, m, r = slice_index[i], slice_index[i + 1], slice_index[i + 2]
                x_mask = torch.zeros((B, N, 1), dtype=x.dtype, device=x.device)
                x_mask[:, l: r, :] = 1.0
                adj_mask = torch.zeros_like(adj)
                adj_mask[:, l: m, m: r] = 1.0
                upper = self.inter_layer[f'{i + 1}_{i}'](x, adj, x_mask=x_mask, adj_mask=adj_mask)
                feat_list.append(upper)

            feat = torch.stack(feat_list, dim=-2)
            B, N, L, C = feat.shape
            feat = torch.reshape(feat, (B * N, L, C))
            feat = self.layer_aggr[f'{i}'](x=torch.reshape(x, (B * N, 1, -1)), k=feat, v=feat)  # (B, N, C)
            feat = torch.reshape(feat, (B, N, C)) * x_mask
            y += feat  # inplace

        y += self.lin_root(x)
        return y


class HeterogeneousGrapherV3(BaseModule):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.aggr1 = HeterogeneousAggregatorV3(in_channels=in_channels,
                                               out_channels=hidden_channels,
                                               n_layers=n_layers)

        self.aggr2 = HeterogeneousAggregatorV3(in_channels=hidden_channels,
                                               out_channels=out_channels,
                                               n_layers=n_layers)
        self.lin1 = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.LayerNorm(out_channels * 4),
            nn.GELU()
        )
        self.lin2 = nn.Sequential(
            nn.Linear(out_channels * 4, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )

    def forward(self, x, edge, slice_index: List[int]):
        identity = x

        x = self.aggr1(x, edge, slice_index)
        x = F.gelu(x)
        x = self.aggr2(x, edge, slice_index)
        x = F.gelu(x)

        x = identity + x
        identity = x

        x = self.lin1(x)
        x = self.lin2(x)

        x = identity + x
        return x
