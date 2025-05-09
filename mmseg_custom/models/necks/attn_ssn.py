import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmseg.models.builder import NECKS

from .core.attention import AttentionGrapher
from .core.layer import HLayer


@NECKS.register_module()
class AttnHSSN(BaseModule):
    def __init__(self, in_channels: List[int], hidden_channels, out_channels: List[int], spix_index: List[int],
                 n_spix: List[int], n_iter, pos_scale, color_scale, fuse_channels: int, freeze_kernel=False,
                 up_index: List[int] = None):
        super().__init__()
        assert len(in_channels) == len(out_channels)

        self.up_index = up_index if up_index is not None else spix_index

        for i in self.up_index:
            assert i in spix_index, f'up_index should be a subset of spix_index'

        self.layers = nn.ModuleList(
            [HLayer(in_channels[spix_index[i]], hidden_channels, out_channels[spix_index[i]],
                    n_spix[i], n_iter, pos_scale, color_scale)
             for i in range(len(spix_index))])

        self.expands_node = nn.ModuleList([nn.Linear(out_channels[i], fuse_channels) for i in spix_index])
        self.reduces_node = nn.ModuleList([nn.Linear(fuse_channels, out_channels[i]) for i in self.up_index])

        self.attn_block = AttentionGrapher(fuse_channels, fuse_channels)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.spix_index = spix_index
        self.n_spix = n_spix
        self.pos_scale = pos_scale
        self.color_scale = color_scale
        self.fuse_channels = fuse_channels
        # self.fuse_resolution = fuse_resolution
        self.gamma_node = nn.ParameterList(
            [nn.Parameter(torch.tensor([1.]), requires_grad=True) for _ in self.up_index])
        self.freeze_kernel = freeze_kernel

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
        if not isinstance(x, list):
            x = list(x)
        node_feat_list = []
        # hard_code_list = []
        total_n_pixel = 0
        affine_list = []
        slice_index = [0]
        for i in range(len(self.spix_index)):
            x_index = self.spix_index[i]
            node_feat, hard_code, fact_n_pixel, affine = self.layers[i](x[x_index])

            node_feat = self.expands_node[i](node_feat)
            node_feat_list.append(node_feat)

            # hard_code = torch.unsqueeze(hard_code, dim=1)
            # hard_code = F.interpolate(hard_code.float(), size=self.fuse_resolution, mode='nearest')
            # hard_code = torch.squeeze(hard_code, dim=1)

            hard_code += total_n_pixel  # broadcast
            # hard_code = torch.unsqueeze(hard_code, dim=1)
            # hard_code_list.append(hard_code)

            total_n_pixel += fact_n_pixel
            slice_index.append(total_n_pixel)

            affine_list.append(affine)

        node_feat = torch.cat(node_feat_list, dim=1)
        # hard_code = torch.cat(hard_code_list, dim=1)

        # adj = spixel3d_to_adj_maxtrix(hard_code, total_n_pixel).float()

        node_feat = self.attn_block(node_feat)

        for i in range(len(self.up_index)):
            x_index = self.up_index[i]
            B, C, H, W = x[x_index].shape
            node_feat_slice = node_feat[:, slice_index[i]:slice_index[i + 1], :]

            node_feat_slice = self.reduces_node[i](node_feat_slice)

            feat = affine_list[i] @ (node_feat_slice * self.gamma_node[i])
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            x[x_index] = feat + x[x_index]
        return x
