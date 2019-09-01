from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .resnet_fpn import ResNet_FPN
from .resnet_fpn_v2 import ResNet_FPN_v2
from .stn_head import STNHead
from .tps_spatial_transformer import TPSSpatialTransformer
from .attention_recognition_head import AttentionRecognitionHead
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss

from examples.config import get_args

global_args = get_args(sys.argv[1:])

__factory = {
    'ResNet_FPN': ResNet_FPN,
    'ResNet_FPN_v2': ResNet_FPN_v2,
}


class ModelBuilder(nn.Module):
    """
    rectificaiton baseline
    """

    def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels,
                 REC_ON=True,
                 FEAT_FUSE=False,
                 tps_margins=(0, 0), STN_ON=False):
        super(ModelBuilder, self).__init__()

        self.arch = arch
        self.rec_num_classes = rec_num_classes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.REC_ON = REC_ON
        self.STN_ON = STN_ON

        self.base = create(self.arch, num_layers=global_args.num_layers)
        base_out_planes = self.base.out_planes

        self.rec_head = AttentionRecognitionHead(
            num_classes=rec_num_classes,
            in_planes=base_out_planes,
            sDim=sDim,
            attDim=attDim,
            max_len_labels=max_len_labels)
        self.rec_crit = SequenceCrossEntropyLoss()

        self.tps = TPSSpatialTransformer(
            output_image_size=tuple(global_args.tps_outputsize),
            num_control_points=global_args.sampling_num_per_side * 2,
            margins=tps_margins)
        self.stn_head = STNHead(
            in_planes=base_out_planes,
            num_ctrlpoints=global_args.sampling_num_per_side * 2,
            activation=global_args.stn_activation)

    def forward(self, input_dict):
        return_dict = {}
        return_dict['losses'] = {}
        return_dict['output'] = {}
        return_dict['raw_centerlines'] = {}

        x, rec_targets, rec_lengths, sym_targets, ctrl_points, sample_mask = input_dict['images'], \
            input_dict['rec_targets'], \
            input_dict['rec_lengths'], \
            input_dict['sym_targets'], \
            input_dict['ctrl_points'], \
            input_dict['mask_flags']

        share_feat = self.base(x)

        if self.training:

            # predict control points
            pred_ctrl_points = self.stn_head(share_feat)  # (x, y)
            # rectification
            rectified_feat, _ = self.tps(share_feat, pred_ctrl_points)

            # recognition
            rec_pred = self.rec_head(
                [rectified_feat, rec_targets, rec_lengths])

            loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
            return_dict['losses']['loss_rec'] = loss_rec
        else:
            # predict control points
            pred_ctrl_points = self.stn_head(share_feat)
            # rectification
            rectified_feat, _ = self.tps(share_feat, pred_ctrl_points)
            return_dict['output']['ctrl_points'] = pred_ctrl_points  # NxTx2

            # recognition
            rec_pred, rec_pred_scores = self.rec_head.sample(
                [rectified_feat, rec_targets, rec_lengths])
            rec_pred_ = self.rec_head(
                [rectified_feat, rec_targets, rec_lengths])

            loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
            return_dict['losses']['loss_rec'] = loss_rec
            return_dict['output']['pred_rec'] = rec_pred

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            return_dict['losses'][k] = v.unsqueeze(0)

        return return_dict


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]
