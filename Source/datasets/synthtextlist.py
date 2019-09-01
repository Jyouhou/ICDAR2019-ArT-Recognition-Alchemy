from __future__ import absolute_import

from .augmentor import color_augment
from Source.utils.labelmaps import get_vocabulary
from .utils import TextSquare, Rotation

import cv2
import os
import sys

import numpy as np
from numpy.linalg import norm
import math
import random
import string
import json

import torch
import torch.utils.data as data

from examples.config import get_args
global_args = get_args(sys.argv[1:])


numericalphabet = list(string.digits + string.ascii_letters)

class SynthTextList(data.Dataset):
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
        super(SynthTextList, self).__init__()
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
        # the ratio between inputs and feature maps
        self.ratio = int(height / 16)
        # args for tps transformation gt.
        self.sampling_num_per_side = global_args.sampling_num_per_side
        # self.postprocessor = Postprocessor()
        self.postprocessor = None

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
        self.BANNED = ("'", '"', ',', '.')

        self.rec_num_classes = len(self.voc)
        self.lowercase = (voc_type.find('LOWERCASE') >= 0)
        # self.lowercase = True

        label_json_path = os.path.join(self.root, 'Label.json')
        labels = json.load(open(label_json_path, 'r'))
        for label in labels:
            nid = os.path.join(self.root, label["img"]) # i.e. image path
            word = label["word"]
            chars = np.transpose(np.array(label["chars"]), (2, 1, 0)) # 2x4xn -> n x 4 x 2
            if self.sample_filter(chars, word, 0.3):
                self.nid_labels.append((nid, word, chars, True))
            else:
                continue
            if len(word) > self.max_len:
                self.max_len = len(word)
            self.counter += 1
            if self.counter >= self.num_samples:
                break

        # the last <eos> should be included.
        self.max_len += 1

        random.shuffle(self.nid_labels)

        self.nSamples = len(self.nid_labels)
        print('\n ===== ===== ===== =====\nread {} images from {}.\n ===== ===== ===== =====\n'.format(
            self.nSamples, self.root))

    def __getitem__(self, indices):
        if isinstance(indices, list):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        assert index <= len(self), 'index range error'
        nid, word, char_box, trans = self.nid_labels[index]
        if trans:
            char_box = char_box.T.reshape((-1, 8))
        else:
            char_box = char_box.reshape((-1, 8))
        img = self.fetcher(nid)

        # remove the margin
        try:
            img, char_box = self.reduce_margin(img, char_box)
            # char_bax: nx8, [x,y,x,y,x,y,x,y]
        except BaseException:
            print(img.shape)

        # data augmentation
        # raw_h and raw_w should be associated with the size which char_box ->
        # img
        if self.is_aug:
            img = color_augment(img)
        img = img[:, :, (2, 1, 0)]  # to rgb
        img = img.astype(np.float)
        raw_h, raw_w = img.shape[0], img.shape[1]
        if global_args.REC_SQUARE == 0:
            img = cv2.resize(img, (self.width, self.height)).astype(np.float64)
            ratio = np.array([[[self.width / raw_w,
                                self.height / raw_h]]])  # 1x1x2
        else:
            img, ratio = TextSquare(img.astype(
                np.float64), global_args.REC_SQUARE)
            img = Rotation(img)
        char_box = char_box.astype(np.float)
        char_box[:, ::2] *= ratio[0, 0, 0]
        char_box[:, 1::2] *= ratio[0, 0, 1]

        img_copy = img.copy()
        img = (img - 128.0) / 128.0  # (img - mean) / std. [-1, 1]
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        flag = True

        if global_args.STN_ON:
            # generate the geometric attributes.
            if global_args.REC_SQUARE == 0:
                map_width, map_height = int(
                    self.width / self.ratio), int(self.height / self.ratio)
            else:
                map_width, map_height = global_args.REC_SQUARE, global_args.REC_SQUARE
            resized_h, resized_w, _ = img.shape
            char_box[:, ::2] = 1.0 * map_width / resized_w * \
                char_box[:, ::2]  # x 在feature map上的位置, 1~64
            char_box[:, 1::2] = 1.0 * map_height / resized_h * \
                char_box[:, 1::2]  # y 在feature map上的位置, 1~16
            # word_rect = cv2.minAreaRect(char_box.reshape(-1, 2))
            # word_box = cv2.boxPoints(word_rect)

            char_box = char_box.reshape((-1, 4, 2))  # x, y
            # word_box = word_box.reshape((-1, 4, 2))
            chars = [
                char_box[i] for i in range(
                    len(char_box)) if word[i] in numericalphabet]

            ctrl_pts_x = np.linspace(
                0.0, (map_width - 1), self.sampling_num_per_side)
            ctrl_pts_y_top = np.ones(self.sampling_num_per_side) * 0.0
            ctrl_pts_y_bottom = np.ones(
                self.sampling_num_per_side) * (map_height - 1)
            ctrl_pts_top = np.stack(
                [ctrl_pts_y_top, ctrl_pts_x], axis=1)
            ctrl_pts_bottom = np.stack(
                [ctrl_pts_y_bottom, ctrl_pts_x], axis=1)
            ctrl_points = np.concatenate(
                [ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)

        else:
            # compatible with the other datasets, which have the segmentation
            # supervision.
            map_width, map_height = int(
                self.width / self.ratio), int(self.height / self.ratio)

            ctrl_points = np.zeros(
                (self.sampling_num_per_side * 2,
                 2)).astype(
                np.float32)

        maps = np.zeros((1, map_height, map_width), np.float32)
        maps = torch.from_numpy(maps).float()
        # recognition labels
        if self.lowercase:
            word = word.lower()
        # fill with the padding token
        label = np.full((self.max_len,),
                        self.char2id[self.PADDING], dtype=np.int)

        label_list = []
        char_count = -1
        for char_n, char in enumerate(word):
            if char in self.char2id:
                ID = self.char2id[char]
                label_list.append(ID)
            elif char == " ":
                ID = None
            else:
                # add the unknown token
                self.UNKNOWN_COUNT += 1
                if self.UNKNOWN_COUNT < 10:
                    print('{0} is out of vocabulary.'.format(char))
                ID = self.char2id[self.UNKNOWN]
                label_list.append(ID)

        # add a stop token
        label_list.append(self.char2id[self.EOS])
        label[:len(label_list)] = np.array(label_list)

        # label length
        label_len = len(label_list)

        # if use multi-dataset to train
        if global_args.MULTI_TRAINDATA:
            # mask flage
            mask_flag = float(flag)
            return (img, label, label_len, maps, ctrl_points, mask_flag)

        return (img, label, label_len, maps, ctrl_points)

    def __len__(self):
        return self.nSamples

    def sample_filter(self, char_box, word, aspect_ratio):
        # remove the sample if it has char_area 0 in it.
        char_box = char_box.T.reshape((-1, 4, 2))
        char_box = np.clip(char_box, 0, math.inf)
        x_min = np.amin(char_box[:, :, 0], axis=1)
        x_max = np.amax(char_box[:, :, 0], axis=1)
        y_min = np.amin(char_box[:, :, 1], axis=1)
        y_max = np.amax(char_box[:, :, 1], axis=1)
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        area = crop_w * crop_h
        if 0.0 in area:
            return False
        flag = True
        if aspect_ratio < 0.2:
            return False
        if word == "":
            return False
        # filter the word, where the first or last char is in self.BANNED
        if word[0] in self.BANNED:
            word = word[1:]
            if word == '':
                return False
        if word[-1] in self.BANNED:
            word = word[:-1]
            if word == '':
                return False
        if not self.is_clockwise(char_box):
            return False
        return True

    def reduce_margin(self, img, char_box):
        """
        The margin of original images are too large, use this funtion to reduce the margin.

        Args:
            img (cv2 image): [h, w, 3] the input image with large margin
            char_box (numpy array int): [n, 8] the char boxes
        """
        h, w, _ = img.shape
        min_x = np.min(char_box[:, ::2])
        min_y = np.min(char_box[:, 1::2])
        max_x = np.max(char_box[:, ::2])
        max_y = np.max(char_box[:, 1::2])

        max_margin = 2
        margin_left = random.randint(0, max_margin)
        margin_top = random.randint(0, max_margin)
        margin_right = random.randint(0, max_margin)
        margin_bottom = random.randint(0, max_margin)

        min_x = max(min_x - margin_left, 0)
        min_y = max(min_y - margin_top, 0)
        max_x = min(max_x + margin_right, w - 1)
        max_y = min(max_y + margin_bottom, h - 1)

        cropped_img = img[min_y:max_y, min_x:max_x, :]
        char_box[:, ::2] = char_box[:, ::2] - min_x
        char_box[:, 1::2] = char_box[:, 1::2] - min_y

        # # for visualization
        # for i in range(char_box.shape[0]):
        #   cv2.polylines(cropped_img, [char_box[i].reshape(4,1,2)], True, (0, 255, 255))

        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            raise Exception

        return cropped_img, char_box

    def shrink(self, img, char_box):
        """
        The margin of original images are too large, use this funtion to reduce the margin.

        Args:
            img (cv2 image): [h, w, 3] the input image with large margin
            char_box (numpy array int): [n, 8] the char boxes
        """
        h, w, _ = img.shape
        min_x = np.min(char_box[:, ::2])
        min_y = np.min(char_box[:, 1::2])
        max_x = np.max(char_box[:, ::2])
        max_y = np.max(char_box[:, 1::2])

        max_margin = 2
        margin_left = random.randint(-max_margin, 0)
        margin_top = random.randint(-max_margin, 0)
        margin_right = random.randint(-max_margin, 0)
        margin_bottom = random.randint(-max_margin, 0)

        min_x = max(min_x - margin_left, 0)
        min_y = max(min_y - margin_top, 0)
        max_x = min(max_x + margin_right, w - 1)
        max_y = min(max_y + margin_bottom, h - 1)

        cropped_img = img[min_y:max_y, min_x:max_x, :]
        char_box[:, ::2] = char_box[:, ::2] - min_x
        char_box[:, 1::2] = char_box[:, 1::2] - min_y

        # # for visualization
        # for i in range(char_box.shape[0]):
        #   cv2.polylines(cropped_img, [char_box[i].reshape(4,1,2)], True, (0, 255, 255))

        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            raise Exception

        return cropped_img, char_box

    def is_clockwise(self, char_box):
        anchor_box = np.array([1, 10, 10, 10, 10, 20, 1, 20])
        anchor_box = anchor_box.reshape(4, 2)
        vec00 = anchor_box[3, :] - anchor_box[0, :]
        vec01 = anchor_box[2, :] - anchor_box[1, :]
        vec02 = anchor_box[1, :] - anchor_box[0, :]
        vec03 = anchor_box[2, :] - anchor_box[3, :]
        vec10s = char_box[:, 3, :] - char_box[:, 0, :]
        vec11s = char_box[:, 2, :] - char_box[:, 1, :]
        vec12s = char_box[:, 1, :] - char_box[:, 0, :]
        vec13s = char_box[:, 2, :] - char_box[:, 3, :]
        # dot10s = np.dot(vec10s, vec00)
        # dot11s = np.dot(vec11s, vec01)
        # dot12s = np.dot(vec12s, vec02)
        # dot13s = np.dot(vec13s, vec03)
        for i in range(char_box.shape[0]):
            vec10 = vec10s[i]
            vec11 = vec11s[i]
            vec12 = vec12s[i]
            vec13 = vec13s[i]
            if np.dot(
                vec00,
                vec10) < 0 or np.dot(
                vec01,
                vec11) < 0 or np.dot(
                vec02,
                vec12) < 0 or np.dot(
                    vec03,
                    vec13) < 0:
                pass
            else:
                return True
        return False

