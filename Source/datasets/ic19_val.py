from __future__ import absolute_import

import cv2
import os
import pdb
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import random as rand
import re
import pickle
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from Source.utils.data import transforms as T
from Source.utils.labelmaps import get_vocabulary
from .utils import TextSquare, Rotation

from examples.config import get_args

global_args = get_args(sys.argv[1:])


class IC19VAL(data.Dataset):
    real_world = True

    def __init__(
            self,
            root,
            voc_type,
            height,
            width,
            num_samples,
            is_aug,
            mix_data=False):
        super(IC19VAL, self).__init__()
        self.root = root
        self.voc_type = voc_type
        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.sampling_num_per_side = global_args.sampling_num_per_side

        self.fetcher = lambda path:cv2.imread(path)
        self.nid_labels = []
        self.max_len = 0

        # for recognition
        assert voc_type in [
            'LOWERCASE',
            'ALLCASES',
            'ALLCASES_SYMBOLS',
            'LOWERCASE_SYMBOLS']
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.UNKNOWN_COUNT = 0
        self.voc = get_vocabulary(
            voc_type,
            EOS=self.EOS,
            PADDING=self.PADDING,
            UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))

        self.rec_num_classes = len(self.voc)
        self.lowercase = (voc_type.find('LOWERCASE') >= 0)
        self.ratio = 4

        label_json_path = os.path.join(self.root, 'Label.json')
        labels = json.load(open(label_json_path, 'r'))
        for label in labels:
            nid = os.path.join(self.root, label["img"]) # i.e. image path
            word = label["word"]
            self.nid_labels.append((nid, word))
            if len(word) > self.max_len:
                self.max_len = len(word)

        # the last <eos> should be included.
        self.max_len += 1

        self.nSamples = len(self.nid_labels)
        print('\n ===== ===== ===== =====\nread {} images from {}.\n ===== ===== ===== =====\n'.format(self.nSamples,
                                                                                                       self.root))

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        assert index <= len(self), 'index range error'

        nid, word, poly_x, poly_y = self.nid_labels[index]
        has_poly = True

        img = self.fetcher(nid)
        # to gray (if trained only with synth90k).
        if global_args.ToGrey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 2)
            img = np.tile(img, (1, 1, 3))
        raw_h, raw_w = img.shape[0], img.shape[1]
        img = img[:, :, (2, 1, 0)]  # to rgb
        img = img.astype(np.float)
        if global_args.REC_SQUARE == 0:
            img = cv2.resize(img, (self.width, self.height)).astype(np.float64)
        else:
            img, ratio = TextSquare(img.astype(
                np.float64), global_args.REC_SQUARE)
#            img = Rotation(img)
        # char_box = char_box * ratio
        img = (img - 128.0) / 128.0  # (img - mean) / std. [-1, 1]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        if global_args.REC_SQUARE == 0:
            map_width, map_height = int(
                self.width / self.ratio), int(self.height / self.ratio)
        else:
            map_width, map_height = global_args.REC_SQUARE, global_args.REC_SQUARE
        maps = np.ones((1, map_height, map_width), np.float32)
        maps = torch.from_numpy(maps).float()

        skel_points = np.zeros((500, 2), np.float32)
        skel_points.fill(-1)
        skel_points = torch.from_numpy(skel_points).float()

        try:
            assert len(poly_x) >= 4 and len(poly_y) >= 4
            poly_x = np.array(poly_x)
            poly_x = poly_x * ratio[0, 0, 0]
            poly_y = np.array(poly_y)
            poly_y = poly_y * ratio[1, 0, 0]
            ctrl_points = get_ctrl_points(
                poly_x, poly_y, self.sampling_num_per_side * 2)  # y,x
            assert ctrl_points is not None
        except:
            ctrl_points = np.zeros(
                (self.sampling_num_per_side * 2,
                 2)).astype(
                np.float32)

        # recognition labels
        if self.lowercase:
            word = word.lower()
        # fill with the padding token
        label = np.full((self.max_len,),
                        self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        for char in word:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            elif char == " ":
                pass
            else:
                # add the unknown token
                self.UNKNOWN_COUNT += 1
                if self.UNKNOWN_COUNT < 10:
                    print('{0} is out of vocabulary.'.format(char))
                label_list.append(self.char2id[self.UNKNOWN])
        # add a stop token
        label_list = label_list + [self.char2id[self.EOS]]
        label[:len(label_list)] = np.array(label_list)

        if global_args.evaluate_with_lexicon:
            return (img, label, len(label_list), maps, ctrl_points, [])
        else:
            return (img, label, len(label_list), maps, ctrl_points, 1.)

    def __len__(self):
        return self.nSamples


def get_l2_dist(p1, p2):
    '''
    the L2 distance between p1 and p2.
    :param p1: point 1 (2, )
    :param p2: point 2 (2, )
    :return:  float
    '''
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sampling(pts, sampling_num):
    '''

    :param pts: (N, 2)
    :param sampling_num:
    :return: (sampling_num, 2)
    '''
    pts = [pts[0]] + pts  # (N + 1, 2)
    distances = [get_l2_dist(pts[i], pts[i + 1])
                 for i in range(len(pts) - 1)]  # (N, 2)
    distance_cumsum = list(np.cumsum(distances))  # (N, 2)
    pts = pts[1:]  # (N, 2)
    total_dist = distance_cumsum[-1]
    segment_length = total_dist / (sampling_num - 1)

    key_points = []
    key_points.append(pts[0])  # head point must be included.
    # sampling sampling_num - 2 points.
    for keypoint_idx in range(1, sampling_num - 1):
        length = keypoint_idx * segment_length
        for i in range(len(pts) - 1):
            if (length > distance_cumsum[i]
                    and length <= distance_cumsum[i + 1]):
                length_to_left = length - distance_cumsum[i]
                length_to_right = distance_cumsum[i + 1] - length
                w0 = length_to_right / (length_to_left + length_to_right)
                w1 = 1.0 - w0
                interpolated_point = (
                    w0 * pts[i][0] + w1 * pts[i + 1][0],
                    w0 * pts[i][1] + w1 * pts[i + 1][1],
                )
                key_points.append(interpolated_point)
                break

    key_points.append(pts[-1])
    if len(key_points) != sampling_num:
        x = rand.random()
        print(
            'id={x:.5}',
            'key point num={}, sampling num={}, skel_points={}'.format(
                len(key_points),
                sampling_num,
                pts))
        return None  # special signal

    return key_points


def get_ctrl_points(poly_x, poly_y, sampling_num):
    '''

    :param poly_x: width, list(int)
    :param poly_y: height, list(int)
    :param sampling_num: the total number of sampling control points. int
    :return:
    '''
    ctrl_pts_x_top = poly_x[:len(poly_x) // 2]
    ctrl_pts_x_bottom = poly_x[len(poly_x) // 2:][::-1]
    ctrl_pts_y_top = poly_y[:len(poly_y) // 2]
    ctrl_pts_y_bottom = poly_y[len(poly_y) // 2:][::-1]

    ctrl_pts_top = [[ctrl_pts_y_top[i], ctrl_pts_x_top[i]]
                    for i in range(len(ctrl_pts_x_top))]
    ctrl_pts_top = sampling(ctrl_pts_top, sampling_num // 2)
    ctrl_pts_bottom = [[ctrl_pts_y_bottom[i], ctrl_pts_x_bottom[i]]
                       for i in range(len(ctrl_pts_x_bottom))]
    ctrl_pts_bottom = sampling(ctrl_pts_bottom, sampling_num // 2)

    if ctrl_pts_top is None or ctrl_pts_bottom is None:
        return None

    return np.array(ctrl_pts_top + ctrl_pts_bottom, dtype=np.float32)
