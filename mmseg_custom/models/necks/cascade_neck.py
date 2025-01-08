from typing import List

from mmseg.models.builder import NECKS
from mmseg.models import builder

from torch import nn


@NECKS.register_module()
class CascadeNeck(nn.Module):

    def __init__(self, neck: List, *args, **kwargs):
        super().__init__()
        self.neck = nn.Sequential()
        for nc in neck:
            self.neck.append(builder.build_neck(nc))

    def forward(self, inputs):
        inputs = self.neck(inputs)
        return inputs
