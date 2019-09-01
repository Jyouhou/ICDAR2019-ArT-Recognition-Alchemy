# --------------------------------------------------------
# Pytorch Faster R-CNN and FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen, Yixiao Ge
# https://github.com/yxgeee/pytorch-FPN/blob/master/lib/nets/resnet_v1.py
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


__all__ = [
    'ResNet_FPN_v3',
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BuildBlock(nn.Module):
    def __init__(self, planes=256):
        super(BuildBlock, self).__init__()

        self.planes = planes
        # Top-down layers, use nn.ConvTranspose2d to replace
        # nn.Conv2d+F.upsample?
        self.toplayer1 = nn.Conv2d(
            2048,
            planes,
            kernel_size=1,
            stride=1,
            padding=0)  # Reduce channels
        self.toplayer2 = nn.Conv2d(
            256, planes, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(
            256, planes, kernel_size=3, stride=1, padding=1)
        self.toplayer4 = nn.Conv2d(
            256, planes, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            1024, planes, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(
            512, planes, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(
            256, planes, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(
            x,
            size=(
                H,
                W),
            mode='bilinear',
            align_corners=True) + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = self.toplayer2(p4)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = self.toplayer3(p3)
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = self.toplayer4(p2)

        return p2, p3, p4, p5


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 32
        # Conv1 (32, 128)
        self.conv1_1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv1_3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.bn1_3 = nn.BatchNorm2d(64)
        self.relu1_3 = nn.ReLU(inplace=True)
        # residual blocks
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=(
                2, 2))  # (16, 64)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=(
                2, 2))  # (8, 32)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=(
                2, 1))  # (4, 32)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=(
                2, 1))  # (2, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ResNet_FPN_v3(nn.Module):
    def __init__(self, num_layers=50):
        super(ResNet_FPN_v3, self).__init__()
        self._num_layers = num_layers
        self._layers = {}

        self._init_head_tail()
        self.out_planes = self.fpn.planes

    def forward(self, x):
        c2 = self.head1(x)
        c3 = self.head2(c2)
        c4 = self.head3(c3)
        c5 = self.head4(c4)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        net_conv = [p2, p3, p4, p5]

        # return p2, [x, self.resnet.conv1(x), c2]
        return p2

    def _init_head_tail(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self.resnet = resnet50()

        elif self._num_layers == 101:
            self.resnet = resnet101()

        elif self._num_layers == 152:
            self.resnet = resnet152()

        else:
            # other numbers are not supported
            raise NotImplementedError

        # Build Building Block for FPN
        self.fpn = BuildBlock()
        self.head1 = nn.Sequential(
            self.resnet.conv1_1,
            self.resnet.bn1_1,
            self.resnet.relu1_1,
            self.resnet.conv1_2,
            self.resnet.bn1_2,
            self.resnet.relu1_2,
            self.resnet.layer1)  # /2, /2
        self.head2 = nn.Sequential(self.resnet.layer2)  # /4, /4
        self.head3 = nn.Sequential(self.resnet.layer3)  # /8, /4
        self.head4 = nn.Sequential(self.resnet.layer4)  # /16, /4
