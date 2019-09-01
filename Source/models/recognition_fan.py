from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class convBlock_basic(nn.Module):
    def __init__(
            self,
            inChannel,
            outChannel,
            kernel,
            stride,
            pad,
            use_batchnorm=False):
        super(convBlock_basic, self).__init__()

        self.use_batchnorm = use_batchnorm

        self.conv = nn.Conv2d(
            inChannel,
            outChannel,
            kernel,
            stride=stride,
            padding=pad)
        if self.use_batchnorm:
            self.bn = nn.BatchNorm2d(outChannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class BiLSTM(nn.Module):
    def __init__(
            self,
            inputDim,
            outDim,
            depth=1,
            bidirectional=True,
            squeeze=False,
            transpose=None,
            flatten=False):
        # flatten 是否展成二维向量(bxc)
        '''
          x: batch, seq, feature

        '''
        super(BiLSTM, self).__init__()
        self.hiddenDim = outDim
        self.flatten = flatten
        self.lstm = nn.LSTM(
            inputDim,
            self.hiddenDim,
            depth,
            bidirectional=bidirectional,
            batch_first=True)  # inputDim, outDim, depth?
        self.lstm_forward = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.lstm_backward = nn.Linear(self.hiddenDim, self.hiddenDim)
        self.transpose = transpose
        self.squeeze = squeeze

        self.reset()
        # self.lstm.flatten_parameters()

    def forward(self, x):
        if self.squeeze:
            x = torch.squeeze(x)
        if self.transpose is not None:
            x = x.transpose(self.transpose[0], self.transpose[1])

        batchSize, seq_len = x.size(0), x.size(1)

        # init state not batch fisrt???
        h0 = torch.zeros(2, batchSize, self.hiddenDim)
        c0 = torch.zeros(2, batchSize, self.hiddenDim)

        # batch, seq_len, hidden_size * num_directions
        output, _ = self.lstm(x, (h0, c0))

        output_forward = output[:, :, :self.hiddenDim]
        output_forward = output_forward.resize(
            output_forward.size(0) * output_forward.size(1), self.hiddenDim)

        output_backward = output[:, :, self.hiddenDim:]
        output_backward = output_backward.resize(
            output_backward.size(0) * output_backward.size(1), self.hiddenDim)

        x = self.lstm_forward(output_forward) + \
            self.lstm_backward(output_backward)
        x = x.view(batchSize, seq_len, -1)

        if self.flatten:
            x = x.view(batchSize, -1)

        return x

    def reset(self):  # re-initial parameters
        # lstm
        for item in dir(self.lstm):
            if item.startswith('weight'):
                # print('init weight: %s ...' %(item))
                init.uniform_(getattr(self.lstm, item), -0.08, 0.08)
            elif item.startswith('bias'):
                if item == 'bias':
                    continue
                else:
                    # print('init bias: %s ...' %(item))
                    init.constant_(getattr(self.lstm, item), 0)

        # fc
        init.uniform_(self.lstm_forward.weight, -0.08, 0.08)
        init.constant_(self.lstm_forward.bias, 0)

        init.uniform_(self.lstm_backward.weight, -0.08, 0.08)
        init.constant_(self.lstm_backward.bias, 0)

        # print('LSTM init finished ...')


class FANet(nn.Module):
    def __init__(self, blockType=BasicBlock):
        super(FANet, self).__init__()

        self.encoder = nn.Sequential()
        # cnn
        self.inplanes = 64
        self.cnn_layers = nn.Sequential()

        # stage 1
        self.s1_conv1 = convBlock_basic(
            inChannel=3,
            outChannel=32,
            kernel=3,
            stride=1,
            pad=1,
            use_batchnorm=True)
        self.s1_conv2 = convBlock_basic(
            inChannel=32,
            outChannel=64,
            kernel=3,
            stride=1,
            pad=1,
            use_batchnorm=True)

        # stage 2
        self.s2_maxpool = nn.MaxPool2d(2, 2)
        self.s2_resblock = self._make_layer(blockType, 128, 1)
        self.s2_conv = convBlock_basic(
            inChannel=128,
            outChannel=128,
            kernel=3,
            stride=1,
            pad=1,
            use_batchnorm=True)

        # stage 3
        self.s3_maxpool = nn.MaxPool2d(2, 2)
        self.s3_resblock = self._make_layer(blockType, 256, 2)
        self.s3_conv = convBlock_basic(
            inChannel=256,
            outChannel=256,
            kernel=3,
            stride=1,
            pad=1,
            use_batchnorm=True)

        # stage 4
        self.s4_maxpool = nn.MaxPool2d(2, stride=(2, 1), padding=(0, 1))
        self.s4_resblock = self._make_layer(blockType, 512, 5)
        self.s4_conv = convBlock_basic(
            inChannel=512,
            outChannel=512,
            kernel=3,
            stride=1,
            pad=1,
            use_batchnorm=True)

        # stage 5
        self.s5_resblock = self._make_layer(blockType, 512, 3)
        self.s5_conv1 = convBlock_basic(
            inChannel=512, outChannel=512, kernel=2, stride=(
                2, 1), pad=(
                0, 1), use_batchnorm=True)
        self.s5_conv2 = convBlock_basic(
            inChannel=512,
            outChannel=512,
            kernel=2,
            stride=1,
            pad=0,
            use_batchnorm=True)

        self.cnn_layers.add_module('s1_conv1', self.s1_conv1)
        self.cnn_layers.add_module('s1_conv2', self.s1_conv2)

        self.cnn_layers.add_module('s2_maxpool', self.s2_maxpool)
        self.cnn_layers.add_module('s2_resblock', self.s2_resblock)
        self.cnn_layers.add_module('s2_conv', self.s2_conv)

        self.cnn_layers.add_module('s3_maxpool', self.s3_maxpool)
        self.cnn_layers.add_module('s3_resblock', self.s3_resblock)
        self.cnn_layers.add_module('s3_conv', self.s3_conv)

        self.cnn_layers.add_module('s4_maxpool', self.s4_maxpool)
        self.cnn_layers.add_module('s4_resblock', self.s4_resblock)
        self.cnn_layers.add_module('s4_conv', self.s4_conv)

        self.cnn_layers.add_module('s5_resblock', self.s5_resblock)
        self.cnn_layers.add_module('s5_conv1', self.s5_conv1)
        self.cnn_layers.add_module('s5_conv2', self.s5_conv2)

        del self.s1_conv1
        del self.s1_conv2
        del self.s2_maxpool
        del self.s2_resblock
        del self.s2_conv
        del self.s3_maxpool
        del self.s3_resblock
        del self.s3_conv
        del self.s4_maxpool
        del self.s4_resblock
        del self.s4_conv
        del self.s5_resblock
        del self.s5_conv1
        del self.s5_conv2

        # rnn
        self.lstm1 = BiLSTM(
            512,
            256,
            bidirectional=True,
            flatten=False,
            squeeze=True,
            transpose=(
                2,
                1))
        self.lstm2 = BiLSTM(
            256,
            256,
            bidirectional=True,
            flatten=False,
            squeeze=False,
            transpose=None)  # bx26x256

        self.encoder.add_module('cnn_layers', self.cnn_layers)
        self.encoder.add_module('lstm1', self.lstm1)
        self.encoder.add_module('lstm2', self.lstm2)

        self.out_planes = 256

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
