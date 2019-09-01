from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
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


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 reduce=True):
        super(MaskedSmoothL1Loss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target, map_mask, sample_mask):
        '''
        param input: [N, C, H, W]
        param target: [N, C, H, W]
        param map_mask: [N, H, W]
        param sample_mask: [N]
        '''
        _assert_no_grad(target)
        # _assert_no_grad(map_mask)
        loss = F.smooth_l1_loss(input, target, reduce=False)
        map_mask = map_mask.unsqueeze(1).expand_as(loss)
        if sample_mask is not None:
            new_shape = [-1]
            for i in range(loss.dim() - 1):
                new_shape.append(1)
            sample_mask = sample_mask.view(
                new_shape).expand_as(loss).type_as(loss)
            map_mask = map_mask * sample_mask
        if False:  # DEBUG
            if torch.isnan(loss).sum() > 0:
                print(
                    'loss nan, num: {0}, {1}'.format(
                        torch.isnan(loss).sum(),
                        torch.isnan(target).sum()))
                nan_mask = torch.isnan(target).nonzero()
                print(nan_mask)
        masked_loss = loss * map_mask
        if self.size_average:
            output = torch.sum(masked_loss) / torch.sum(map_mask)
        elif self.reduce:
            output = torch.sum(masked_loss)
        else:
            output = masked_loss
        return output
