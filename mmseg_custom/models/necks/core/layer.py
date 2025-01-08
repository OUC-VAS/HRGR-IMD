import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ops_ssn.modules import calc_init_centroid

from .kernel import Kernel, KernelWOXY


class HLayer(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_spix, n_iter, pos_scale, color_scale):
        super().__init__()
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel = Kernel(in_channels, hidden_channels, n_spix, n_iter)

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.n_spix

        inputs = x

        nspix_per_axis = int(math.sqrt(N))
        pos_scale = self.pos_scale * max(nspix_per_axis / H, nspix_per_axis / W)

        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'),
            0)
        coords = coords[None].repeat(B, 1, 1, 1).float()

        inputs = torch.cat([self.color_scale * inputs, pos_scale * coords], 1)
        coords = coords * pos_scale
        Q, hard, dist_feat = self.kernel(inputs, coords)

        affine_cluster = Q.transpose(1, 2)

        if affine_cluster.is_sparse:
            affine_cluster = affine_cluster.to_dense().contiguous()

        affine_cluster = affine_cluster / (affine_cluster.sum(dim=1, keepdims=True) + 1e-8)

        node_feat = (x.reshape(B, C, H * W) @ affine_cluster).permute(0, 2, 1)

        fact_n_pixel = Q.shape[1]

        affine = torch.arange(fact_n_pixel, device=x.device).reshape(1, 1, -1).repeat(B, H * W, 1) == hard.unsqueeze(-1)
        affine = affine.float()

        hard = hard.reshape(B, H, W)

        return node_feat, hard, fact_n_pixel, affine


class HLayerWOXY(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_spix, n_iter, pos_scale, color_scale):
        super().__init__()
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel = KernelWOXY(in_channels, hidden_channels, n_spix, n_iter)

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.n_spix

        inputs = x

        # nspix_per_axis = int(math.sqrt(N))

        Q, hard, dist_feat = self.kernel(inputs)

        affine_cluster = Q.transpose(1, 2)

        if affine_cluster.is_sparse:
            affine_cluster = affine_cluster.to_dense().contiguous()

        affine_cluster = affine_cluster / (affine_cluster.sum(dim=1, keepdims=True) + 1e-8)

        node_feat = (x.reshape(B, C, H * W) @ affine_cluster).permute(0, 2, 1)

        fact_n_pixel = Q.shape[1]

        affine = torch.arange(fact_n_pixel, device=x.device).reshape(1, 1, -1).repeat(B, H * W, 1) == hard.unsqueeze(-1)
        affine = affine.float()

        hard = hard.reshape(B, H, W)

        return node_feat, hard, fact_n_pixel, affine

class SSNLayer(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_spix, n_iter, pos_scale, color_scale, soft_affine):
        super().__init__()
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel = Kernel(in_channels, hidden_channels, n_spix, n_iter)
        self.expand = nn.Linear(hidden_channels, out_channels)
        self.soft_affine = soft_affine

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.n_spix

        inputs = x

        nspix_per_axis = int(math.sqrt(N))
        pos_scale = self.pos_scale * max(nspix_per_axis / H, nspix_per_axis / W)

        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'),
            0)
        coords = coords[None].repeat(B, 1, 1, 1).float()

        inputs = torch.cat([self.color_scale * inputs, pos_scale * coords], 1)
        coords = coords * pos_scale
        Q, hard, dist_feat = self.kernel(inputs, coords)

        affine = torch.arange(N, device=x.device).reshape(1, 1, -1).repeat(B, H * W, 1) == hard.unsqueeze(-1)
        affine = affine.float()

        if self.soft_affine:
            affine_cluster = affine / (affine.sum(dim=1, keepdims=True) + 1e-8)
        else:
            affine_cluster = affine
        node_feat = (x.reshape(B, C, H * W) @ affine_cluster).permute(0, 2, 1)
        dist_feat = self.expand(dist_feat.permute(0, 2, 1))

        fact_n_pixel = Q.shape[1]
        hard = hard.reshape(B, H, W)

        return node_feat, dist_feat, hard, fact_n_pixel, affine


class HGLayer(BaseModule):
    def __init__(self, n_spix):
        super().__init__()
        self.n_spix = n_spix

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.n_spix

        num_spixels_width = int(math.sqrt(N * W / H))
        num_spixels_height = int(math.sqrt(N * H / W))

        node_feat, hard = calc_init_centroid(x, num_spixels_width, num_spixels_height)
        hard = hard.int()

        node_feat = node_feat.transpose(1, 2)

        fact_n_pixel = num_spixels_width * num_spixels_height

        affine = torch.arange(fact_n_pixel, device=x.device).reshape(1, 1, -1).repeat(B, H * W, 1) == hard.unsqueeze(-1)
        affine = affine.float()

        hard = hard.reshape(B, H, W)

        return node_feat, hard, fact_n_pixel, affine
