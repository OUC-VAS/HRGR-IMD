import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class NonLocalMask(nn.Module):
    def __init__(self, in_channels, reduce_scale):
        super(NonLocalMask, self).__init__()

        self.r = reduce_scale

        # input channel number
        self.ic = in_channels * self.r * self.r

        # middle channel number
        self.mc = self.ic

        self.g = nn.Conv2d(in_channels=self.ic, out_channels=self.ic,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.ic, out_channels=self.mc,
                             kernel_size=1, stride=1, padding=0)

        self.W_s = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.W_c = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                             kernel_size=1, stride=1, padding=0)

        self.gamma_s = nn.Parameter(torch.ones(1))

        self.gamma_c = nn.Parameter(torch.ones(1))

        self.get_mask = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            value :
                f: B X (HxW) X (HxW)
                ic: intermediate channels
                z: feature maps( B X C X H X W)
            output:
                mask: feature maps( B X 1 X H X W)
        """

        b, c, h, w = x.shape

        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)

        # g x
        g_x = self.g(x1).view(b, self.ic, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta
        theta_x = self.theta(x1).view(b, self.mc, -1)

        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x

        # phi x
        phi_x = self.phi(x1).view(b, self.mc, -1)

        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        # non-local attention
        f_s = torch.matmul(theta_x_s, phi_x_s)
        f_s_div = F.softmax(f_s, dim=-1)

        f_c = torch.matmul(theta_x_c, phi_x_c)
        f_c_div = F.softmax(f_c, dim=-1)

        # get y_s
        y_s = torch.matmul(f_s_div, g_x)
        y_s = y_s.permute(0, 2, 1).contiguous()
        y_s = y_s.view(b, c, h, w)

        # get y_c
        y_c = torch.matmul(g_x, f_c_div)
        y_c = y_c.view(b, c, h, w)

        # get z
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)

        # get mask
        mask = torch.sigmoid(self.get_mask(z.clone()))

        return mask, z


@HEADS.register_module()
class NLCHead(BaseDecodeHead):
    def __init__(self, crop_size=(256, 256), **kwargs):
        super(NLCHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert self.num_classes == 2
        self.crop_size = crop_size

        feat1_num, feat2_num, feat3_num, feat4_num = self.in_channels

        self.get_mask4 = NonLocalMask(feat4_num, 1)
        self.get_mask3 = NonLocalMask(feat3_num, 2)
        self.get_mask2 = NonLocalMask(feat2_num, 2)
        self.get_mask1 = NonLocalMask(feat1_num, 4)

    def _forward_feature(self, inputs):
        """
            inputs :
                feat : a list contains features from s1, s2, s3, s4
            output:
                mask1: output mask ( B X 1 X H X W)
                pred_cls: output cls (B X 4)
        """
        s1, s2, s3, s4 = inputs

        s1 = F.interpolate(s1, size=self.crop_size, mode='bilinear', align_corners=self.align_corners)
        s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode='bilinear', align_corners=self.align_corners)
        s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode='bilinear', align_corners=self.align_corners)
        s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode='bilinear', align_corners=self.align_corners)

        mask4, z4 = self.get_mask4(s4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode='bilinear', align_corners=self.align_corners)

        s3 = s3 * mask4U
        mask3, z3 = self.get_mask3(s3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode='bilinear', align_corners=self.align_corners)

        s2 = s2 * mask3U
        mask2, z2 = self.get_mask2(s2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        s1 = s1 * mask2U
        mask1, z1 = self.get_mask1(s1)

        return z1, mask2, mask3, mask4

    def forward(self, inputs):
        z1, mask2, mask3, mask4 = self._forward_feature(inputs)
        output = self.cls_seg(z1)
        return output, mask2, mask3, mask4

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        output, mask2, mask3, mask4 = self.forward(inputs)
        losses1 = self.losses(output, gt_semantic_seg)
        losses2 = self.losses(mask2, gt_semantic_seg)
        losses3 = self.losses(mask3, gt_semantic_seg)
        losses4 = self.losses(mask4, gt_semantic_seg)
        losses = dict()
        losses['acc_seg'] = losses1['acc_seg']
        losses['loss_ce'] = losses1['loss_ce'] + losses2['loss_ce'] + losses3['loss_ce'] + losses4['loss_ce']
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)[0]


def make_mask_balance(mask):
    mask_balance = torch.ones_like(mask)
    if (mask == 1).sum():
        mask_balance[mask == 1] = 0.5 / ((mask == 1).sum().to(torch.float) / mask.numel())
        mask_balance[mask == 0] = 0.5 / ((mask == 0).sum().to(torch.float) / mask.numel())
    else:
        # print('Mask balance is not working!')
        pass
    return mask_balance
