from typing import List

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import nn

from .graph import DenseSAGEConv, CrossAttention


class HeterogeneousAggregatorV2(BaseModule):

    def __init__(self, in_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intra_layer = nn.ModuleDict()
        self.inter_layer = nn.ModuleDict()
        self.n_layer = n_layers
        for i in range(self.n_layer):
            self.intra_layer[f'{i}'] = DenseSAGEConv(in_channels, out_channels, normalize=True)

        for i in range(self.n_layer - 1):
            self.inter_layer[f'{i}_{i + 1}'] = DenseSAGEConv(in_channels, out_channels, normalize=True)
            self.inter_layer[f'{i + 1}_{i}'] = DenseSAGEConv(in_channels, out_channels, normalize=True)

        self.layer_aggr = nn.ModuleDict()
        for i in range(self.n_layer):
            self.layer_aggr[f'{i}'] = CrossAttention(out_channels)

        self.q_list = nn.ParameterDict()
        for i in range(self.n_layer):
            self.q_list[f'{i}'] = nn.Parameter(torch.rand(1, out_channels))

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
            q = torch.broadcast_to(self.q_list[f'{i}'], (B * N, 1, C))
            feat = self.layer_aggr[f'{i}'](x=q, k=feat, v=feat)  # (B, N, C)
            feat = torch.reshape(feat, (B, N, C)) * x_mask
            y += feat  # inplace
        return y


class HeterogeneousGrapherV2(BaseModule):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.aggr1 = HeterogeneousAggregatorV2(in_channels=in_channels,
                                               out_channels=hidden_channels,
                                               n_layers=n_layers)

        self.aggr2 = HeterogeneousAggregatorV2(in_channels=hidden_channels,
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
