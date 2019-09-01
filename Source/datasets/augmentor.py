import random

import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from scipy.ndimage.filters import gaussian_filter


def quad2minrect(boxes):
    # trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape(
        (-1, 1)), boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))


def quad2boxlist(boxes):
    res = []
    for i in range(boxes.shape[0]):
        res.append([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]], [
                   boxes[i][4], boxes[i][5]], [boxes[i][6], boxes[i][7]]])
    return res


def boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1]
                           [1], box[2][0], box[2][1], box[3][0], box[3][1]])
    return res


def rotate_polygons(polygons, angle, r_c):
    # polygons: N*8
    # r_x: rotate center x
    # r_y: rotate center y
    # angle: -15~15

    poly_list = quad2boxlist(polygons)
    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords)) < 5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = boxlist2quads(rotate_boxes_list)
    return res


class Augment:
    @classmethod
    def augment(self, method_name, im, *args, **kwds):
        self.__getattribute__(method_name)(self, im, *args, **kwds)

    @classmethod  # rgb
    def gaussian_blur(self, im):
        kernel = int(round(random.normalvariate(5, 5))) * 2 + 1
        sigma = int(round(random.uniform(1, 5))) * 2 + 1
        return cv2.GaussianBlur(im, (kernel, kernel), sigma)

    @classmethod  # hsv
    def saturation(self, im):
        lower = 0.5
        upper = 1.5

        assert upper >= lower, "saturation upper must be >= lower."
        assert lower >= 0, "saturation lower must be non-negative."
        im[:, :, 1] *= random.uniform(lower, upper)
        return im

    @classmethod  # hsv
    def hue(self, im):
        delta = 18
        im[:, :, 2] += random.uniform(-delta, delta)
        im[:, :, 2][im[:, :, 2] > 360.0] -= 360.0
        im[:, :, 2][im[:, :, 2] < 0.0] += 360.0
        return im

    @classmethod  # rgb
    def lighting_noise(self, im):
        perms = ((0, 1, 2), (0, 2, 1),
                 (1, 0, 2), (1, 2, 0),
                 (2, 0, 1), (2, 1, 0))

        swap = perms[random.randint(0, len(perms) - 1)]
        im = im[:, :, swap]
        return im

    @classmethod  # ?
    def contrast(self, im):
        lower = 0.5
        upper = 1.5
        assert upper >= lower, "contrast upper must be >= lower."
        assert lower >= 0, "contrast lower must be non-negative."
        alpha = random.uniform(lower, upper)
        im *= alpha
        return im

    @classmethod  # rgb
    def brightness(self, im):
        lower = 0.8
        upper = 1.1
        dtype = im.dtype
        C = random.uniform(lower, upper)
        im = np.clip(im * C, 0, 255).astype(dtype)
        return im

    @classmethod  # rgb
    def linear_motionblur(self, im):
        def _gen_kernel(angle, length):
            rad = np.deg2rad(angle)

            dx = np.cos(rad)
            dy = np.sin(rad)
            a = int(max(list(map(abs, (dx, dy)))) * length * 2)
            kern = np.zeros((a, a))
            cx, cy = int(a / 2), int(a / 2)
            dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
            cv2.line(kern, (cx, cy), (dx, dy), 1.0)

            s = kern.sum()
            if s == 0:
                kern[cx, cy] = 1.0
            else:
                kern /= s

            return kern
        angle = random.uniform(0, 360)
        length = random.randint(1, 5)
        kern = _gen_kernel(angle, length)
        im = cv2.filter2D(im, -1, kern)
        return im

    @classmethod
    def rotate_image(self, im):
        delta = 30
        delta = random.uniform(-1 * delta, delta)
        # rotate image first
        height, width, _ = im.shape
        # get the minimal rect to cover the rotated image
        img_box = np.array([[0, 0, width, 0, width, height, 0, height]])
        rotated_img_box = quad2minrect(rotate_polygons(
            img_box, -1 * delta, (width / 2, height / 2)))
        r_height = int(max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(
            rotated_img_box[0][3], rotated_img_box[0][1]))
        r_width = int(max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(
            rotated_img_box[0][2], rotated_img_box[0][0]))

        # padding im
        start_h, start_w = max(
            0, int((r_height - height) / 2.0)), max(0, int((r_width - width) / 2.0))
        end_h, end_w = start_h + height, start_w + width
        im_padding = np.zeros(
            (max(r_height, end_h), max(r_width, end_w), 3))
        im_padding[start_h:end_h, start_w:end_w, :] = im

        M = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), delta, 1)
        im = cv2.warpAffine(im_padding, M, (r_width, r_height))

        return im


def color_augment(img):
    augmentor = Augment()
    rand_ = np.random.rand()
    if rand_ < 0.7:
        img = augmentor.gaussian_blur(img)
    # rand_ = np.random.rand()
    # if rand_ < 0.2:
    #     img = augmentor.saturation(img)
    # rand_ = np.random.rand()
    # if rand_ < 0.2:
    #     img = augmentor.hue(img)
    rand_ = np.random.rand()
    if rand_ < 0.5:
        img = augmentor.lighting_noise(img)
    # rand_ = np.random.rand()
    # if rand_ < 0.1:
    #     img = augmentor.contrast(img)
    rand_ = np.random.rand()
    if rand_ < 0.8:
        img = augmentor.brightness(img)
    rand_ = np.random.rand()
    if rand_ < 0.4:
        img = augmentor.linear_motionblur(img)
    return img


if __name__ == '__main__':
    img = np.random.random((2, 4, 3)) * 255
    img = img.astype(np.uint8)
    color_augment(img)
