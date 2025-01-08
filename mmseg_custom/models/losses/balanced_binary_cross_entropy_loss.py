import torch
from mmseg.models import LOSSES
from torch import nn
import torch.nn.functional as F


@LOSSES.register_module()
class BalancedBinaryCrossEntropyLoss(nn.Module):

    def __init__(self,
                 loss_name='loss_ce',
                 loss_weight=1.0,
                 use_sigmoid=True,
                 reduction='mean'):
        super(BalancedBinaryCrossEntropyLoss, self).__init__()
        assert use_sigmoid is True
        self._loss_name = loss_name
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid

    def forward(self,
                pred: torch.Tensor,
                label: torch.Tensor,
                **kwargs):
        pos = torch.eq(label, 1).float()
        neg = torch.eq(label, 0).float()
        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg
        alpha_pos = num_neg / num_total
        alpha_neg = num_pos / num_total
        weights = alpha_pos * pos + alpha_neg * neg

        if pred.dim() == 4 and label.dim() == 3:
            if pred.size(1) != 1:
                pred = torch.softmax(pred, dim=1)[:, 1, :, :]
            pred = pred.squeeze(1)
        else:
            raise NotImplementedError
        loss_cls = self.loss_weight * F.binary_cross_entropy(
            pred, label.float(), weights, reduction=self.reduction)
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
