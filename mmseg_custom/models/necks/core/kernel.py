import torch
from mmcv.runner import BaseModule
from ops_ssn.modules import ssn_iter, sparse_ssn_iter
from torch import nn
import torch.nn.functional as F


def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


class Kernel(BaseModule):
    def __init__(self, in_channels, hidden_channels, n_spix, n_iter=10):
        super().__init__()
        self.n_spix = n_spix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(in_channels + 2, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64 * 3 + in_channels + 2, hidden_channels - 2, 3, padding=1),
            nn.ReLU(True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, coords):
        pixel_f = self.feature_extract(x, coords)

        if self.training:
            return ssn_iter(pixel_f, self.n_spix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.n_spix, self.n_iter)

    def feature_extract(self, x, coords):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = F.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return torch.cat([feat, coords], 1)


class KernelWOXY(BaseModule):
    def __init__(self, in_channels, hidden_channels, n_spix, n_iter=10):
        super().__init__()
        self.n_spix = n_spix
        self.n_iter = n_iter

        self.scale1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            conv_bn_relu(64, 64)
        )
        self.scale2 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.scale3 = nn.Sequential(
            nn.MaxPool2d(3, 2, padding=1),
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(64 * 3 + in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pixel_f = self.feature_extract(x)

        if self.training:
            return ssn_iter(pixel_f, self.n_spix, self.n_iter)
        else:
            return sparse_ssn_iter(pixel_f, self.n_spix, self.n_iter)

    def feature_extract(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        s2 = F.interpolate(s2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=s1.shape[-2:], mode="bilinear", align_corners=False)

        cat_feat = torch.cat([x, s1, s2, s3], 1)
        feat = self.output_conv(cat_feat)

        return feat
