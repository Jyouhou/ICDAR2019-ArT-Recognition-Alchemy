from __future__ import print_function, absolute_import

from examples.config import get_args

import os
import time
from time import gmtime, strftime

import cv2
import torch

import numpy as np
import sys

from . import evaluation_metrics
from .utils.meters import AverageMeter
from .utils.visualization_utils import VisTPS, VisIMG

metrics_factory = evaluation_metrics.factory()

global_args = get_args(sys.argv[1:])


class BaseEvaluator(object):
    def __init__(self, model, metric, use_cuda=True):
        super(BaseEvaluator, self).__init__()
        self.model = model
        self.metric = metric
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def evaluate(
            self,
            data_loader,
            step=1,
            print_freq=1,
            tfLogger=None,
            dataset=None,
            vis_dir=None):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        # forward the network
        images, outputs, targets, losses = [], {}, [], []
        file_names = []
        if global_args.evaluate:
            raw_centerlines = []
        else:
            raw_centerlines = None

        end = time.time()

        # ====== Computation ====== #
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            input_dict = self._parse_data(inputs)
            with torch.no_grad():
                output_dict = self._forward(input_dict)

                batch_size = input_dict['images'].size(0)

                total_loss_batch = 0.
                for k, loss in output_dict['losses'].items():
                    loss = loss.mean(dim=0, keepdim=True)
                    total_loss_batch += loss.item() * batch_size

                images.append(input_dict['images'].cpu()[:, :, ::2, ::2])
                targets.append(input_dict['rec_targets'].cpu())
                losses.append(total_loss_batch)
            if global_args.evaluate_with_lexicon:
                file_names += input_dict['file_name']
            for k, v in output_dict['output'].items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v.cpu())
            if raw_centerlines is not None:
                raw_centerlines += output_dict['raw_centerlines']

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('[{}]\t'
                      'Evaluation: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                              i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        images = torch.cat(images)
        targets = torch.cat(targets)
        for k, v in outputs.items():
            outputs[k] = torch.cat(outputs[k])

        # ====== Evaluation ====== #
        if 'pred_rec' in outputs:
            # evaluation with metric
            if global_args.evaluate_with_lexicon:
                eval_res, pred_list = metrics_factory[self.metric + '_with_lexicon'](
                    outputs['pred_rec'], targets, dataset, file_names)
                print(
                    'lexicon0: {0}, {1:.3f}'.format(
                        self.metric, eval_res[0]))
                print(
                    'lexicon50: {0}, {1:.3f}'.format(
                        self.metric, eval_res[1]))
                print(
                    'lexicon1k: {0}, {1:.3f}'.format(
                        self.metric, eval_res[2]))
                print(
                    'lexiconfull: {0}, {1:.3f}'.format(
                        self.metric, eval_res[3]))
                eval_res = eval_res[0]
            else:
                eval_res, pred_list = metrics_factory[self.metric](
                    outputs['pred_rec'], targets, dataset)
                print('lexicon0: {0}: {1:.3f}'.format(self.metric, eval_res))

        else:
            eval_res = None
        # ====== Visualization ====== #
        if vis_dir is not None and os.path.isdir(vis_dir):
            if 'ctrl_points' in outputs:
                # for rectification baselines
                images, rectified_imgs = VisTPS(images, outputs['ctrl_points'])
                for i in range(rectified_imgs.shape[0]):
                    cv2.imwrite(os.path.join(vis_dir, f"{i}-{str(pred_list[i])}.jpg"),
                                np.concatenate([images[i],
                                                rectified_imgs[i][::2, ::2, :]], axis = 0))
            else:
                images = VisIMG(images)
                for i in range(len(images)):
                    cv2.imwrite(os.path.join(vis_dir, f"{i}-{str(pred_list[i])}.jpg"),
                                images[i])


        self.model.train()
        return eval_res

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError


class Evaluator(BaseEvaluator):
    def _parse_data(self, inputs):
        input_dict = {}
        if len(inputs) == 5:
            imgs, label_encs, lengths, maps, ctrl_points = inputs
            mask_flags = None
        elif len(inputs) == 6:  # multi datasets
            if global_args.evaluate_with_lexicon:
                mask_flags = None
                imgs, label_encs, lengths, maps, ctrl_points, file_name = inputs
            else:
                imgs, label_encs, lengths, maps, ctrl_points, mask_flags = inputs

        with torch.no_grad():
            images = imgs.to(self.device)
            if label_encs is not None:
                labels = label_encs.to(self.device)
            if maps is not None:
                maps = maps.to(self.device)
            if ctrl_points is not None:
                ctrl_points = ctrl_points.to(self.device)
            if mask_flags is not None:
                mask_flags = mask_flags.to(self.device)

        input_dict['images'] = images
        input_dict['rec_targets'] = labels
        input_dict['rec_lengths'] = lengths
        input_dict['sym_targets'] = maps
        input_dict['ctrl_points'] = ctrl_points
        input_dict['mask_flags'] = mask_flags
        if global_args.evaluate_with_lexicon:
            input_dict['file_name'] = file_name
        return input_dict

    def _forward(self, input_dict):
        self.model.eval()
        with torch.no_grad():
            output_dict = self.model(input_dict)
        return output_dict
