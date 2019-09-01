from __future__ import absolute_import

import cv2
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import random
import re
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from Source.utils.data import transforms as T
from Source.utils.labelmaps import get_vocabulary
from .utils import TextSquare

from examples.config import get_args
global_args = get_args(sys.argv[1:])


class prediction(data.Dataset):
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
        super(prediction, self).__init__()
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
            word = ""
            self.nid_labels.append((nid, word))

        # the last <eos> should be included.
        self.max_len = 71

        self.nSamples = len(self.nid_labels)
        print('\n ===== ===== ===== =====\nread {} images from {}.\n ===== ===== ===== =====\n'.format(
            self.nSamples, self.root))

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        assert index <= len(self), 'index range error'
        nid, word = self.nid_labels[index]

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
        # char_box = char_box * ratio
        # img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_REPLICATE)
        # img = cv2.resize(img, (self.width, self.height))
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
            return (img, label, len(label_list), maps, ctrl_points, 0.)

    def __len__(self):
        return self.nSamples
