import math
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models.builder import NECKS
from ops_ssn.modules import spixel2d_to_adj_maxtrix, spixel3d_to_adj_maxtrix

from .core.graph import Grapher
from .core.kernel import Kernel
from .core.layer import SSNLayer


@NECKS.register_module()
class SSN(BaseModule):
    def __init__(self, in_channels: List[int], hidden_channels, out_channels: List[int], spix_index: List[int],
                 n_spix, n_iter, pos_scale, color_scale):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.layers = nn.ModuleList(
            [Layer(in_channels[i], hidden_channels, out_channels[i], n_spix[i], n_iter, pos_scale, color_scale)
             for i in spix_index])
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.spix_index = spix_index
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale

    def forward(self, x):
        return [self.layers[i](x[i]) for i in self.spix_index] + x[len(self.spix_index):]


class Layer(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, n_spix, n_iter, pos_scale, color_scale):
        super().__init__()
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel = Kernel(in_channels, hidden_channels, n_spix, n_iter)
        self.expand = nn.Linear(hidden_channels, out_channels)
        self.graph_conv_dist = Grapher(out_channels, out_channels * 2, out_channels)
        self.graph_conv_node = Grapher(out_channels, out_channels * 2, out_channels)
        # self.attn_node = nn.Sequential(
        #     nn.LayerNorm(out_channels),
        #     MultiheadAttention(embed_dims=out_channels, num_heads=4, batch_first=True),
        # )
        # self.attn_dist = nn.Sequential(
        #     nn.LayerNorm(out_channels),
        #     MultiheadAttention(embed_dims=out_channels, num_heads=4, batch_first=True),
        # )

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
        node_feat = (x.reshape(B, C, H * W) @ affine).permute(0, 2, 1)
        dist_feat = self.expand(dist_feat.permute(0, 2, 1))

        fact_n_pixel = Q.shape[1]
        adj = spixel2d_to_adj_maxtrix(hard.reshape(B, H, W).float(), fact_n_pixel)

        adj = adj.float()
        node_feat = self.graph_conv_node(node_feat, adj)
        dist_feat = self.graph_conv_dist(dist_feat, adj)

        feat = affine @ (node_feat + dist_feat)
        feat = feat.transpose(1, 2).reshape(B, C, H, W)
        return feat + x


@NECKS.register_module()
class HSSN(BaseModule):
    def __init__(self, in_channels: List[int], hidden_channels, out_channels: List[int], spix_index: List[int],
                 n_spix: List[int], n_iter, pos_scale, color_scale, fuse_channels: int,
                 fuse_resolution: Tuple[int, int], soft_affine=False, freeze_kernel=False):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.layers = nn.ModuleList(
            [SSNLayer(in_channels[i], hidden_channels, out_channels[i], n_spix[i], n_iter, pos_scale, color_scale,
                      soft_affine)
             for i in spix_index])

        self.expands_dist = nn.ModuleList([nn.Linear(out_channels[i], fuse_channels) for i in spix_index])
        self.reduces_dist = nn.ModuleList([nn.Linear(fuse_channels, out_channels[i]) for i in spix_index])
        self.expands_node = nn.ModuleList([nn.Linear(out_channels[i], fuse_channels) for i in spix_index])
        self.reduces_node = nn.ModuleList([nn.Linear(fuse_channels, out_channels[i]) for i in spix_index])

        self.graph_conv_dist = Grapher(fuse_channels, fuse_channels * 2, fuse_channels)
        self.graph_conv_node = Grapher(fuse_channels, fuse_channels * 2, fuse_channels)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.spix_index = spix_index
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.fuse_channels = fuse_channels
        self.fuse_resolution = fuse_resolution
        self.soft_affine = soft_affine
        self.freeze_kernel = freeze_kernel
        if self.soft_affine:
            self.gamma_dist = nn.ParameterList(
                [nn.Parameter(torch.tensor([.2]), requires_grad=True) for _ in spix_index])
            self.gamma_node = nn.ParameterList(
                [nn.Parameter(torch.tensor([1.]), requires_grad=True) for _ in spix_index])

        if self.freeze_kernel:
            self._freeze_weights('kernel')

    def _freeze_weights(self, name: str):
        from mmseg.utils import get_root_logger
        logger = get_root_logger()
        logger.info(f'Freeze {name} weights for '
                    f'{self.__class__.__name__}')

        for k, p in self.named_parameters():
            if name in k:
                p.requires_grad = False

    def init_weights(self):
        from mmseg.utils import get_root_logger
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            from mmcv.runner import _load_checkpoint
            ckpt = _load_checkpoint(self.init_cfg.checkpoint,
                                    logger=logger,
                                    map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('neck.'):
                    state_dict[k[5:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            meg = self.load_state_dict(state_dict, False)
            logger.info(meg)

    def forward(self, x):
        node_feat_list = []
        dist_feat_list = []
        hard_code_list = []
        total_n_pixel = 0
        affine_list = []
        slice_index = [0]
        for i in self.spix_index:
            node_feat, dist_feat, hard_code, fact_n_pixel, affine = self.layers[i](x[i])

            node_feat = self.expands_node[i](node_feat)
            dist_feat = self.expands_dist[i](dist_feat)
            node_feat_list.append(node_feat)
            dist_feat_list.append(dist_feat)

            hard_code = torch.unsqueeze(hard_code, dim=1)
            hard_code = F.interpolate(hard_code.float(), size=self.fuse_resolution, mode='nearest')
            # hard_code = torch.squeeze(hard_code, dim=1)

            hard_code += total_n_pixel  # broadcast
            # hard_code = torch.unsqueeze(hard_code, dim=1)
            hard_code_list.append(hard_code)

            total_n_pixel += fact_n_pixel
            slice_index.append(total_n_pixel)

            affine_list.append(affine)

        node_feat = torch.cat(node_feat_list, dim=1)
        dist_feat = torch.cat(dist_feat_list, dim=1)
        hard_code = torch.cat(hard_code_list, dim=1)

        adj = spixel3d_to_adj_maxtrix(hard_code, total_n_pixel).float()

        node_feat = self.graph_conv_node(node_feat, adj)
        dist_feat = self.graph_conv_dist(dist_feat, adj)

        result_list = []
        for i in self.spix_index:
            B, C, H, W = x[i].shape
            node_feat_slice = node_feat[:, slice_index[i]:slice_index[i + 1], :]
            dist_feat_slice = dist_feat[:, slice_index[i]:slice_index[i + 1], :]

            node_feat_slice = self.reduces_node[i](node_feat_slice)
            dist_feat_slice = self.reduces_dist[i](dist_feat_slice)

            if self.soft_affine:
                feat = affine_list[i] @ (node_feat_slice * self.gamma_node[i] + dist_feat_slice * self.gamma_dist[i])
            else:
                feat = affine_list[i] @ (node_feat_slice + dist_feat_slice)
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            result_list.append(feat + x[i])

        return [*result_list, *x[len(self.spix_index):]]

    def show_result(self,
                    image=None,
                    mask=None,
                    palette=None,
                    opacity=0.5):
        import numpy as np
        if palette is None:
            classes = mask.max() + 1
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(classes, 3))
            np.random.set_state(state)
        if mask is not None:
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_mask[mask == label, :] = color
            if image is not None:
                image = image * (1 - opacity) + color_mask * opacity
                color_mask = image.astype(np.uint8)
            return color_mask

    def visualize_mask(self, mask):
        import numpy as np
        from PIL import Image
        mask = np.uint8(mask[0].detach().cpu())
        color_mask = Image.fromarray(self.show_result(mask=mask))
        color_mask.show()
