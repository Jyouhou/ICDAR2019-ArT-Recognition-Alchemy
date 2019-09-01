from __future__ import absolute_import

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


def conv_bn_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(
            inplace=True),
    )


class RecSubNet(nn.Module):
    """
    This is the backbone for recognition.
    input: [b x in_planes x 16 x 64]
    output: [b x 31 x 2*conv_out_planes]
    """

    def __init__(self, in_planes):
        super(RecSubNet, self).__init__()
        self.in_planes = in_planes

        self.cnn = nn.Sequential()

        # this structure is just follow fots.
        conv_out_planes = [64, 64, 128, 128, 256, 256, 256]
        self.conv1 = conv_bn_relu(in_planes, conv_out_planes[0], 3, 1, 1)
        self.conv2 = conv_bn_relu(
            conv_out_planes[0], conv_out_planes[1], 3, 1, 1)
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2), (0, 0))  # 8 x 32

        self.conv3 = conv_bn_relu(
            conv_out_planes[1], conv_out_planes[2], 3, 1, 1)
        self.conv4 = conv_bn_relu(
            conv_out_planes[2], conv_out_planes[3], 3, 1, 1)
        self.pool2 = nn.MaxPool2d((2, 1), (2, 1), (0, 0))  # 4 x 32

        self.conv5 = conv_bn_relu(
            conv_out_planes[3], conv_out_planes[4], 3, 1, 1)
        self.conv6 = conv_bn_relu(
            conv_out_planes[4], conv_out_planes[5], 3, 1, 1)
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1), (0, 0))  # 2 x 32
        self.conv7 = conv_bn_relu(
            conv_out_planes[5],
            conv_out_planes[6],
            2,
            1,
            0)  # 1 x 31

        self.cnn.add_module('conv1', self.conv1)
        self.cnn.add_module('conv2', self.conv2)
        self.cnn.add_module('pool1', self.pool1)
        self.cnn.add_module('conv3', self.conv3)
        self.cnn.add_module('conv4', self.conv4)
        self.cnn.add_module('pool2', self.pool2)
        self.cnn.add_module('conv5', self.conv5)
        self.cnn.add_module('conv6', self.conv6)
        self.cnn.add_module('pool3', self.pool3)
        self.cnn.add_module('conv7', self.conv7)

        del self.conv1
        del self.conv2
        del self.pool1
        del self.conv3
        del self.conv4
        del self.pool2
        del self.conv5
        del self.conv6
        del self.pool3
        del self.conv7

        self.rnn = nn.LSTM(
            conv_out_planes[5],
            conv_out_planes[5],
            bidirectional=True,
            num_layers=1,
            batch_first=True)
        self.out_planes = 2 * conv_out_planes[5]

        self.init_weights(self.cnn)

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        cnn_feat = self.cnn(x)
        cnn_feat = cnn_feat.squeeze()  # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)
        # self.rnn.flatten_parameters()
        rnn_feat, _ = self.rnn(cnn_feat)
        return rnn_feat