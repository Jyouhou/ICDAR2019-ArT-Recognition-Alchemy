from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .resnet_fpn import ResNet_FPN
from .resnet_fpn_v2 import ResNet_FPN_v2
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
    Squared input
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

        self.rec_base = create(
            self.arch, num_layers=global_args.num_layers)  # for recognition
#        self.sym_base = create(self.arch, num_layers=global_args.num_layers) # for detection
        # assert self.rec_base.out_planes == self.sym_base.out_planes
        base_out_planes = self.rec_base.out_planes

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

        self.AvgPool = torch.nn.AvgPool2d(2, 2)

    def forward(self, input_dict):
        """
        :param input_dict:
        :return:
        """
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

        # x: 256x256
        if global_args.REC_ON_INPUT:
            loc_img = self.AvgPool(x).detach()  # 128x128

            if self.training:
                pred_ctrl_points = self.stn_head(self.rec_base(loc_img))
                rectified_img, _ = self.tps(
                    x, pred_ctrl_points)  # should be 64x256
                rec_feat = self.rec_base(rectified_img)
                rec_pred = self.rec_head([rec_feat, rec_targets, rec_lengths])
                loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
                return_dict['losses']['loss_rec'] = loss_rec
            else:

                pred_ctrl_points = self.stn_head(self.rec_base(loc_img))
                rectified_img, _ = self.tps(x, pred_ctrl_points)
                rec_feat = self.rec_base(rectified_img)
                rec_pred, rec_pred_scores = self.rec_head.sample(
                    [rec_feat, rec_targets, rec_lengths])
                rec_pred_ = self.rec_head([rec_feat, rec_targets, rec_lengths])
                loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)

                return_dict['output']['ctrl_points'] = pred_ctrl_points
                return_dict['losses']['loss_rec'] = loss_rec
                return_dict['output']['pred_rec'] = rec_pred
                return_dict['output']['pred_rec_score'] = rec_pred_scores
        else:
            features = self.rec_base(x)

            if self.training:
                pred_ctrl_points = self.stn_head(self.AvgPool(features))
                rec_feat, _ = self.tps(
                    features, pred_ctrl_points)  # should be 16x64
                #rec_feat = self.rec_base(rectified_features)
                rec_pred = self.rec_head([rec_feat, rec_targets, rec_lengths])
                loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
                return_dict['losses']['loss_rec'] = loss_rec
            else:

                pred_ctrl_points = self.stn_head(self.AvgPool(features))
                rec_feat, _ = self.tps(features, pred_ctrl_points)
#                rec_feat = self.rec_base(rectified_features)
                rec_pred, rec_pred_scores = self.rec_head.sample(
                    [rec_feat, rec_targets, rec_lengths])
                rec_pred_ = self.rec_head([rec_feat, rec_targets, rec_lengths])
                loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)

                return_dict['output']['ctrl_points'] = pred_ctrl_points
                return_dict['losses']['loss_rec'] = loss_rec
                return_dict['output']['pred_rec'] = rec_pred
                return_dict['output']['pred_rec_score'] = rec_pred_scores

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
            return_dict['losses'][k] = v.unsqueeze(0)

        return return_dict


class STNHead(nn.Module):
    # slightly modified for squarization
    def __init__(self, in_planes, num_ctrlpoints, activation='none'):
        super(STNHead, self).__init__()
        self.Upsample = torch.nn.UpsamplingBilinear2d(size=(32, 32))
        self.in_planes = in_planes
        self.num_ctrlpoints = num_ctrlpoints
        self.activation = activation
        self.stn_convnet = nn.Sequential(
            nn.Conv2d(in_planes, 64, 5, 2, 2),  # 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 * 4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 2*2
        self.stn_fc1 = nn.Sequential(
            nn.Linear(4 * 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))
        self.stn_fc2 = nn.Linear(256, int(num_ctrlpoints * 2))

        self.init_weights(self.stn_convnet)
        self.init_weights(self.stn_fc1)
        self.init_stn(self.stn_fc2)

    def init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def init_stn(self, stn_fc2):
        margin = 0.01
        sampling_num_per_side = int(self.num_ctrlpoints / 2)
        ctrl_pts_x = np.linspace(margin, 1. - margin, sampling_num_per_side)
        ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
        ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1 - margin)
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        ctrl_points = np.concatenate(
            [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
        if self.activation is 'none':
            pass
        elif self.activation == 'sigmoid':
            ctrl_points = -np.log(1. / ctrl_points - 1.)
        stn_fc2.weight.data.zero_()
        stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.Upsample(x)
        x = self.stn_convnet(x)
        batch_size, _, h, w = x.size()
        x = x.view(batch_size, -1)
        x = self.stn_fc1(x)
        x = self.stn_fc2(x)
        if self.activation == 'sigmoid':
            x = F.sigmoid(x)
        x = x.view(-1, self.num_ctrlpoints, 2)
        return x
