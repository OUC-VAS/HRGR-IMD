# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Union

import torch
import torch.nn.functional as F
from mmcls.models.losses import accuracy
from mmseg.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule, auto_fp16

from ..utils.build_layers import build_norm_layer, build_act_layer
from torch import nn


# noinspection PyDefaultArgument
@HEADS.register_module()
class ConvHead(BaseModule):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 cls_scale=1.5,
                 act_layer='GELU',
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
                 ):
        super(ConvHead, self).__init__()

        assert isinstance(loss, dict)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.loss_cls = build_loss(loss)

        self.conv_head = nn.Sequential(
            nn.Conv2d(self.in_channels,
                      int(self.in_channels * cls_scale),
                      kernel_size=1,
                      bias=False),
            build_norm_layer(int(self.in_channels * cls_scale), 'BN',
                             'channels_first', 'channels_first'),
            build_act_layer(act_layer))

        self.head = nn.Linear(int(self.in_channels * cls_scale), num_classes) \
            if num_classes > 0 else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def losses(self, cls_score, gt_label):
        num_samples = len(cls_score)
        loss = dict()
        # compute loss
        loss_val = self.loss_cls(cls_score, gt_label, avg_factor=num_samples)
        loss['loss_ce'] = loss_val
        loss['acc_cls'] = accuracy(cls_score, gt_label)
        return loss

    def forward_train(self, inputs, gt_label):
        cls_logits = self.forward(inputs)
        losses = self.losses(cls_logits, gt_label)
        return losses

    # noinspection PyMethodMayBeStatic
    def pre_logits(self, x):
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[-1]
        return x

    @auto_fp16()
    def forward(self, x):
        x = self.pre_logits(x)

        x = self.conv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)

        return x

    def forward_test(self, inputs, softmax=True):
        """Inference without augmentation.

        Args:
            inputs (Tensor): The input classification score logits.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            softmax (bool): Whether to softmax the classification score.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        cls_score = self.forward(inputs)
        if softmax:
            pred = (F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        return pred
