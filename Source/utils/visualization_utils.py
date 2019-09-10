from __future__ import absolute_import

import matplotlib
matplotlib.use('Agg')

from examples.config import get_args
from ..models.tps_spatial_transformer import TPSSpatialTransformer
from . import to_numpy
import sys


global_args = get_args(sys.argv[1:])


def VisTPS(
        images,
        ctrl_points):

    tps = TPSSpatialTransformer(
        output_image_size=(64, 256),
        num_control_points=global_args.sampling_num_per_side * 2,
        margins=global_args.tps_margins).cuda()

    tps_images, _ = tps(images.cuda(), ctrl_points.cuda())

    images = images.permute(0, 2, 3, 1)
    images = to_numpy(images)
    images = images * 128.0 + 128.0
    tps_images = tps_images.permute(0, 2, 3, 1)
    tps_images = to_numpy(tps_images)
    tps_images = tps_images * 128.0 + 128.0

    return images, tps_images

def VisIMG(
        images
        ):

    images = images.permute(0, 2, 3, 1)
    images = to_numpy(images)
    images = images * 128.0 + 128.0

    return images

