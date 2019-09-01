from __future__ import absolute_import
import sys
sys.path.append('./')


from examples.config import get_args
from .utils import TextSquare, Rotation
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import string
import re
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import json

from Source.utils.data import transforms as T
from Source.utils.labelmaps import get_vocabulary
from .augmentor import color_augment



global_args = get_args(sys.argv[1:])


numericalphabet = list(string.digits + string.ascii_letters)


class Synth90k(data.Dataset):
    real_world = False

    def __init__(
            self,
            root,
            voc_type,
            height,
            width,
            num_samples,
            is_aug,
            mix_data=False):
        super(Synth90k, self).__init__()
        self.root = root
        self.voc_type = voc_type
        self.height = height
        self.width = width
        self.num_samples = num_samples
        self.is_aug = is_aug
        self.fetcher = lambda path:cv2.imread(path)
        self.nid_labels = []
        self.max_len = 0
        self.counter = 0
        self.sampling_num_per_side = global_args.sampling_num_per_side

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
            self.counter += 1
            if self.counter >= num_samples:
                break

        # the last <eos> should be included.
        self.max_len += 1

        # random.shuffle(self.nid_labels)
        self.nSamples = len(self.nid_labels)
        print('\n ===== ===== ===== =====\nread {} images from {}.\n ===== ===== ===== =====\n'.format(
            self.nSamples, self.root))

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        assert index <= len(self), 'index range error'
        nid, word, trans = self.nid_labels[index]

        img = self.fetcher(nid)
        if self.is_aug:
            img = color_augment(img)
        raw_h, raw_w = img.shape[0], img.shape[1]
        img = img[:, :, (2, 1, 0)]  # to rgb
        img = img.astype(np.float)
        if global_args.REC_SQUARE == 0:
            img = cv2.resize(img, (self.width, self.height)).astype(np.float64)
        else:
            img, ratio = TextSquare(img.astype(
                np.float64), global_args.REC_SQUARE)
            img = Rotation(img)
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

        ctrl_points = np.zeros(
                (2 * self.sampling_num_per_side,
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

        # label length
        label_len = len(label_list)

        # if use multi-dataset to train
        if global_args.MULTI_TRAINDATA:
                # mask flage
            mask_flag = 0.
            return (img, label, label_len, maps, ctrl_points, mask_flag)

        return (img, label, label_len, maps, ctrl_points)

    def __len__(self):
        return self.nSamples
