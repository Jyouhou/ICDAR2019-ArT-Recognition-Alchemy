import cv2
import random
import numpy as np
import sys

from examples.config import get_args
global_args = get_args(sys.argv[1:])

def TextSquare(img, size):
    """
    :param img: np.float, HxWx3
    :param size: int
    :return:
    """
    h, w, _ = img.shape
    aspect_ratio_augment = random.uniform(0.7, 1.3)
    if w > h:
        target_w = size + 0
        target_h = min(target_w, int(target_w * h / w * aspect_ratio_augment))
    else:
        target_h = size + 0
        target_w = min(target_h, int(target_h * w / h * aspect_ratio_augment))

    ratio = np.array([[[target_w / w,
                        target_h / h]]])  # 2x1x1

    # resize
    resized_img = cv2.resize(img, (target_w, target_h)).astype(np.float64)

    # pad
    pad_h = max(size - target_h, 0)
    pad_w = max(size - target_w, 0)
    pad_left_h = 0 * random.randint(0, pad_h)
    pad_left_w = 0 * random.randint(0, pad_w)
    pad_size = ((pad_left_h, pad_h - pad_left_h),
                (pad_left_w, pad_w - pad_left_w),
                (0, 0))
    padded_img = np.pad(resized_img,
                        pad_size,
                        'constant',
                        constant_values=128.)

    return padded_img, ratio


def Rotation(img, *args, **kwargs):
    probability = global_args.RotationInTraining
    p = random.random()
    if p < 1 - probability:
        return img
    if p < 1 - 2 / 3 * probability:
        degree = 90
    elif p < 1 - 1 / 3 * probability:
        degree = 180
    else:
        degree = 270
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
