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


class MaskedSTNLoss(nn.Module):
    def __init__(self,
                 weight=None, mode='l2'):
        super(MaskedSTNLoss, self).__init__()
        self.weight = weight
        self.mode = mode
        assert mode in ['l1', 'l2']

    def forward(self, input, target, sample_mask):
        '''
        param input: [N, K, 2], pred_ctrl_points
        param target: [N, K, 2], source_ctrl_points
        param sample_mask: [N]
        '''
        _assert_no_grad(target)
        assert input.size(1) == target.size(1)

        if self.mode == 'l1':
            loss_stn_reg = F.l1_loss(input, target, reduce=False)
        elif self.mode == 'l2':
            loss_stn_reg = F.mse_loss(input, target, reduce=False)

        stn_mask = sample_mask.view(-1,
                                    1,
                                    1).expand_as(loss_stn_reg).type_as(loss_stn_reg)
        loss_stn_reg = loss_stn_reg * stn_mask

        # size average
        loss_stn_reg = torch.sum(loss_stn_reg) / \
            torch.sum(sample_mask).type_as(loss_stn_reg)

        return loss_stn_reg
