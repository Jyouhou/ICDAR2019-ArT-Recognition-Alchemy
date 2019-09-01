from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')

import math
import argparse
import os.path as osp


parser = argparse.ArgumentParser()

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# data set
parser.add_argument('--train_dataset', type=str, default='synth90k',
                    help='type of datasets')
parser.add_argument(
    '--train_data_dir',
    type=str,
    metavar='PATH',
    default='./dataset/synth90k/')
parser.add_argument(
    '--extra_train_dataset',
    nargs='+',
    type=str,
    default=[],
    help='type of datasets')
parser.add_argument(
    '--extra_train_data_dir',
    nargs='+',
    type=str,
    metavar='PATH',
    default=[])
parser.add_argument('--test_dataset', type=str, default='ic19_val')
parser.add_argument(
    '--test_data_dir',
    type=str,
    metavar='PATH',
    default='./dataset/ic19_val_test')
parser.add_argument('--MULTI_TRAINDATA', action='store_false', default=True,
                    help='whether use the extra_train_data for training.')

parser.add_argument(
    '--ToGrey',
    action='store_true',
    default=False,
    help='If trained with snth90k only')

# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
# optimization & training setting
parser.add_argument('-b', '--batch_size', type=int, default=512)
parser.add_argument('-j', '--workers', type=int, default=8)
parser.add_argument(
    '--height',
    type=int,
    default=64,
    help="input height, default: 256 for resnet*, "
    "64 for inception")
parser.add_argument(
    '--width',
    type=int,
    default=256,
    help="input width, default: 128 for resnet*, "
    "256 for inception")
parser.add_argument(
    '--voc_type',
    type=str,
    default='LOWERCASE_SYMBOLS',
    choices=[
        'LOWERCASE',
        'ALLCASES',
        'ALLCASES_SYMBOLS',
        'LOWERCASE_SYMBOLS'])
# As synthetic datasets have a large size, reading all data maybe less efficient when debugging. Set a small number and save some time.
parser.add_argument('--num_train', type=int, default=math.inf)
parser.add_argument('--num_test', type=int, default=99999)
# Set as True to apply color jitters to synthetic data when training (deprecated, not in use)
parser.add_argument('--aug', action='store_true', default=False,
                    help='whether use data augmentation.')
# Weighted sampling for real word. The multiplier for the "number" of real word
parser.add_argument('--real_multiplier', type=int, default=10)

# model
parser.add_argument('-a', '--arch', type=str, default='ResNet_FPN')
parser.add_argument('-nl', '--num_layers', type=int,
                    default=50, choices=[50, 101, 152])
parser.add_argument('--sampling_num_per_side', type=int, default=10,
                    help='the number of ctrl points per side.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--REC_ON', action='store_false', default=True,
                    help='add the recognition head.')
parser.add_argument('--STN_ON', action='store_false', default=True,
                    help='add the stn head.')
parser.add_argument('--CRNN', action='store_true', default=False,
                    help='CRNN baseline.')
parser.add_argument('--tps_margins', nargs='+', type=float, default=[0, 0])
parser.add_argument('--stn_activation', type=str, default='sigmoid')
parser.add_argument('--tps_outputsize', nargs='+', type=int, default=[16, 64])
parser.add_argument(
    '--REC_ON_INPUT',
    action='store_true',
    default=False,
    help='the input to recognition net is raw image or hidden features.')
parser.add_argument(
    '--REC_SQUARE',
    default=0,
    type=int,
    help='')

# lstm
parser.add_argument('--decoder_sdim', type=int, default=256,
                    help="the dim of hidden layer in decoder.")
parser.add_argument('--attDim', type=int, default=256,
                    help="the dim for attention.")
# optimizer
parser.add_argument('--lr', type=float, default=1,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
# the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1, 1, 1])
parser.add_argument('--finetune_lr', type=float, default=0.1,
                    help='learning rate at the fine-tune stage.')
parser.add_argument('--finetune_epoch', type=int, default=1,
                    help='epochs of the fine-tune stage.')

# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--start_save', type=int, default=0,
                    help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--cuda', default=True, type=bool,
                    help='whether use cuda support.')
parser.add_argument('--RotationInTraining', type=float, default=0)


# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy',
                    choices=["accuracy", "accuracy_with_lexicon","editdistance", "editdistance_with_lexicon"])
parser.add_argument(
    '--evaluate_with_lexicon',
    action='store_true',
    default=False)
# misc
working_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
parser.add_argument('--logs_dir', type=str, metavar='PATH',
                    default=osp.join(working_dir, 'logs'))
parser.add_argument('--debug', action='store_true',
                    help="if debugging, some steps will be passed.")
parser.add_argument('--vis_dir', type=str, metavar='PATH', default='',
                    help="whether visualize the results while evaluation.")


def get_args(sys_args):
    global_args = parser.parse_args(sys_args)
    global_args.num_control_points = global_args.sampling_num_per_side * 2
    if global_args.REC_ON_INPUT:
        global_args.tps_outputsize = [64, 256]
    return global_args
