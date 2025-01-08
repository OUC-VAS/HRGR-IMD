from typing import List, Tuple, Dict, TypeVar

import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import nn
from torch.nn import Linear


class DenseSAGEConv(torch.nn.Module):
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
        self.lin_root = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

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
        out = self.lin_rel(out) + self.lin_root(x)

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


class Grapher(BaseModule):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.conv1 = DenseSAGEConv(in_channels=in_channels,
                                   out_channels=hidden_channels,
                                   normalize=True)

        self.conv2 = DenseSAGEConv(in_channels=hidden_channels,
                                   out_channels=out_channels,
                                   normalize=True)
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

    def forward(self, x, edge):
        identity = x

        x = self.conv1(x, edge)
        x = F.gelu(x)
        x = self.conv2(x, edge)
        x = F.gelu(x)

        x = identity + x
        identity = x

        x = self.lin1(x)
        x = self.lin2(x)

        x = identity + x
        return x


class CrossAttention(nn.Module):
    r""" Cross Attention Module
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        attn_head_dim (int, optional): Dimension of attention head.
        out_dim (int, optional): Dimension of output.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None,
                 q_dim=None,
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim if q_dim is None else q_dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads,
                      -1).permute(2, 0, 3, 1,
                                  4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1,
                                                             4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HeterogeneousAggregator(BaseModule):

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
            self.layer_aggr[f'{i}'] = CrossAttention(out_channels, q_dim=in_channels)

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
        return y



def dense_to_sparse(adj) -> torch.Tensor:
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    assert 2 == adj.dim()
    return adj.nonzero().t().contiguous()


class HeterogeneousGrapher(BaseModule):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.aggr1 = HeterogeneousAggregator(in_channels=in_channels,
                                             out_channels=hidden_channels,
                                             n_layers=n_layers)

        self.aggr2 = HeterogeneousAggregator(in_channels=hidden_channels,
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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



KeyType = TypeVar("KeyType", str, Tuple[Tuple[str, str, str]])


# def unbind_batch(data: Dict[KeyType, torch.Tensor]) -> List[Dict[KeyType, torch.Tensor]]:
#     unbind: List[Dict[KeyType, torch.Tensor]] = list()
#     assert len(data) != 0, 'data should not be empty'
#     for key, x in data:
#         B = x.shape[0]
#         break
#     # noinspection PyUnboundLocalVariable
#     for _ in range(B):
#         unbind.append(dict())
#
#     for key, x in data:
#         x_list = x.unbind(0)
#         for i in range(B):
#             unbind[i][key] = x_list[i]
#     return unbind


def bind_batch(data: List[Dict[KeyType, torch.Tensor]]) -> Dict[KeyType, torch.Tensor]:
    B = len(data)
    data_dict: Dict[KeyType, torch.Tensor] = dict()
    keys = data[0].keys()
    for k in keys:
        torch_list = []
        for i in range(B):
            torch_list.append(data[i][k])

        data_dict[k] = torch.stack(torch_list, dim=0)
    return data_dict

