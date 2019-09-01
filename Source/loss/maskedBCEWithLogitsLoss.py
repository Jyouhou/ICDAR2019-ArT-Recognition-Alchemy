from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='elementwise_mean',
                 pos_weight=None):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, input, target, sample_mask):
        '''
        param input: [N, H, W]
        param target: [N, H, W]
        param sample_mask: [N]
        '''
        _assert_no_grad(target)
        loss = F.binary_cross_entropy_with_logits(
            input, target, reduce=False)  # [N, H, W]
        if sample_mask is not None:
            sample_mask = sample_mask.view(-1, 1, 1).expand_as(loss)
        else:
            sample_mask = torch.Tensor(loss.size()).fill_(1)
        sample_mask = sample_mask.type_as(loss)
        masked_loss = loss * sample_mask
        if self.size_average:
            output = torch.sum(masked_loss) / torch.sum(sample_mask)
        elif self.reduce:
            output = torch.sum(masked_loss)
        else:
            output = masked_loss
        return output
