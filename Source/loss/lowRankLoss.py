from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


class LowRankLoss(nn.Module):
    """docstring for LowRankLoss"""

    def __init__(self,
                 tol=0.0,
                 margin=0,
                 size_average=True,
                 reduce=True,
                 reverse=False):
        super(LowRankLoss, self).__init__()
        # like numpy.linalg.matrix_rank, tol is the threshold
        self.tol = tol
        self.margin = margin
        self.size_average = size_average
        self.reduce = reduce
        self.reverse = reverse

    def forward(self, raw_feat, rectified_feat):
        """
        get the rank of input tensors.
        :param raw_feat: tensor [n, c, h, w]
        :param rectified_feat: tensor [n, c, h, w]
        """
        assert raw_feat.size() == rectified_feat.size()
        batch_size, _, h, w = raw_feat.size()
        # average alone the channel
        raw_feat = torch.mean(raw_feat, dim=1, keepdim=False)  # [n, h, w]
        rectified_feat = torch.mean(
            rectified_feat, dim=1, keepdim=False)  # [n, h, w]
        # l2 normalize the feat
        flatten_raw_feat = raw_feat.view(batch_size, -1)  # [n, h*w]
        flatten_rectified_feat = rectified_feat.view(
            batch_size, -1)  # [n, h*w]
        raw_feat_n = torch.norm(
            flatten_raw_feat,
            p=2,
            dim=1,
            keepdim=True)  # [n, 1]
        rectified_feat_n = torch.norm(
            flatten_rectified_feat, p=2, dim=1, keepdim=True)  # [n, 1]
        # get the normalized feat
        normed_raw_feat = flatten_raw_feat.div(raw_feat_n).view(-1, h, w)
        normed_rectified_feat = flatten_rectified_feat.div(
            rectified_feat_n).view(-1, h, w)
        # get the rank of normalized feat
        batch_s1, batch_s2 = [], []
        for i in range(batch_size):
            u1, s1, v1 = torch.svd(normed_raw_feat[i])
            u2, s2, v2 = torch.svd(normed_rectified_feat[i])
            batch_s1.append(s1)
            batch_s2.append(s2)
        batch_s1 = torch.stack(batch_s1, dim=0)  # [b, min(h, w)]
        batch_s2 = torch.stack(batch_s2, dim=0)  # [b, min(h, w)]
        # keep the singular value > the tol
        batch_s1 = torch.gt(batch_s1, self.tol)
        batch_s2 = torch.gt(batch_s2, self.tol)
        # get the rank after clamping
        rank1 = batch_s1.sum(dim=1, keepdim=False)  # [b]
        rank2 = batch_s2.sum(dim=1, keepdim=False)  # [b]
        # get the loss
        y = rank1.new()
        y.resize_as_(rank1)
        if self.reverse:
            y.fill_(-1)
        else:
            y.fill_(1)
        loss = F.margin_ranking_loss(
            rank1,
            rank2,
            y,
            margin=self.margin,
            size_average=False,
            reduce=False)
        # loss / length of each word
        if self.size_average:
            loss = torch.sum(loss) / batch_size
        elif self.reduce:
            loss = torch.sum(loss)
        else:
            pass
        return loss
