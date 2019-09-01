from __future__ import absolute_import

import sys
sys.path.append('./')

from config import get_args
from Source import datasets
from Source.datasets.concatdataset import ConcatDataset
from Source.trainers import Trainer
from Source.evaluators import Evaluator
from Source.utils.logging import Logger, TFLogger
from Source.utils.serialization import load_checkpoint
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

import time
import math
import numpy as np
import os.path as osp
import os


global_args = get_args(sys.argv[1:])
if global_args.REC_SQUARE > 0: # Squarization
    from Source.models.Squarization import ModelBuilder
elif global_args.REC_ON_INPUT: # must come with --tps_outputsize 64 256 
    from Source.models.RecInput import ModelBuilder # STN + rec on image
elif global_args.CRNN: # The non-rect baseline
    from Source.models.CRNN_Baseline import ModelBuilder
else:  # rectification baseline
    from Source.models.RectificationBaseline import ModelBuilder


def get_data(name,
             data_dir,
             voc_type,
             height,
             width,
             batch_size,
             workers,
             num_samples,
             is_train,
             is_aug):
    """
    :param name: dataset name/type
    :param data_dir: dataset path
    :param others, as suggested by its var_name
    :return:
    """
    if isinstance(name, list):
        assert isinstance(data_dir, list)
        dataset_list = []
        for _name, _data_dir in zip(name, data_dir):
            dataset_list.append(datasets.create(_name,
                                                _data_dir,
                                                voc_type,
                                                height,
                                                width,
                                                num_samples,
                                                is_aug))
        dataset = ConcatDataset(dataset_list, global_args.real_multiplier)
    else:
        dataset = datasets.create(
            name,
            data_dir,
            voc_type,
            height,
            width,
            num_samples,
            is_aug)

    if is_train:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

    return dataset, data_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    args.cuda = args.cuda and torch.cuda.is_available()

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        train_tfLogger = TFLogger(osp.join(args.logs_dir, 'train'))
        eval_tfLogger = TFLogger(osp.join(args.logs_dir, 'eval'))

    # Save the args to disk
    if not args.evaluate:
        cfg_save_path = osp.join(args.logs_dir, 'cfg.txt')
        cfgs = vars(args)
        with open(cfg_save_path, 'w') as f:
            for k, v in cfgs.items():
                f.write('{}: {}\n'.format(k, v))

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (64, 256)

    if not args.evaluate:
        # multi train data by default
        args.train_data_dir = [args.train_data_dir] + args.extra_train_data_dir
        args.train_dataset = [args.train_dataset] + args.extra_train_dataset

        train_dataset, train_loader = \
            get_data(args.train_dataset, args.train_data_dir, args.voc_type,
                     args.height, args.width, args.batch_size, args.workers, args.num_train, True, args.aug)
    test_dataset, test_loader = \
        get_data(args.test_dataset, args.test_data_dir, args.voc_type,
                 args.height, args.width, args.batch_size, args.workers, args.num_test, False, False)

    if args.evaluate:
        max_len = test_dataset.max_len
    else:
        max_len = max(train_dataset.max_len, test_dataset.max_len)
        train_dataset.max_len = test_dataset.max_len = max_len
    # Create model
    model = ModelBuilder(arch=args.arch,
                         rec_num_classes=test_dataset.rec_num_classes,
                         sDim=args.decoder_sdim,
                         attDim=args.attDim,
                         max_len_labels=max_len,
                         REC_ON=args.REC_ON,
                         FEAT_FUSE=False,
                         tps_margins=tuple(args.tps_margins),
                         STN_ON=args.STN_ON)

    # Load from checkpoint
    if args.evaluation_metric == 'accuracy':
        best_res = 0
    elif args.evaluation_metric == 'editdistance':
        best_res = math.inf
    else:
        raise NotImplementedError(
            "Unsupported evaluation metric:",
            args.evaluation_metric)
    start_epoch = 0
    start_iters = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # use different tps margins in testing stage.
        if hasattr(model, 'tps'):
            filtered_states = {
                k: v for k,
                v in checkpoint['state_dict'].items() if not k.startswith('tps')}
            filtered_states['tps.inverse_kernel'] = model.tps.inverse_kernel
            filtered_states['tps.padding_matrix'] = model.tps.padding_matrix
            filtered_states['tps.target_coordinate_repr'] = model.tps.target_coordinate_repr
            filtered_states['tps.target_control_points'] = model.tps.target_control_points
            model.load_state_dict(filtered_states)
        else:
            model.load_state_dict(checkpoint['state_dict'])

        # compatibility with the epoch-wise evaluation version
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch']
        else:
            start_iters = checkpoint['iters']
            start_epoch = int(start_iters //
                              len(train_loader)) if not args.evaluate else 0
        best_res = checkpoint['best_res']
        print("=> Start iters {}  best res {:.1%}"
              .format(start_iters, best_res))

    if args.cuda:
        device = torch.device("cuda")
        model = model.to(device)
        model = nn.DataParallel(model)

    # Evaluator
    evaluator = Evaluator(model, args.evaluation_metric, args.cuda)

    if args.evaluate:
        print('Test on {0}:'.format(args.test_dataset))
        if len(args.vis_dir) > 0:
            vis_dir = osp.join(args.logs_dir, args.vis_dir)
            if not osp.exists(vis_dir):
                os.makedirs(vis_dir)
        else:
            vis_dir = None

        start = time.time()
        evaluator.evaluate(
            test_loader,
            dataset=test_dataset,
            vis_dir=vis_dir)
        print(f"Finished-{args.test_dataset}")
        print('it took {0} s.'.format(time.time() - start))
        return

    # Optimizer
    param_groups = model.parameters()
    param_groups = filter(lambda p: p.requires_grad, param_groups)
    optimizer = optim.Adadelta(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 4, 5], gamma=0.1)

    # Trainer
    loss_weights = {}
    loss_weights['loss_sym_cls'] = args.loss_weights[0]
    loss_weights['loss_sym_reg'] = args.loss_weights[1]
    loss_weights['loss_rec'] = args.loss_weights[2]
    loss_weights['loss_stn_reg'] = 1.
    loss_weights['sum_seg_loss'] = 1.
    if args.debug:
        args.print_freq = 1
    trainer = Trainer(
        model,
        args.evaluation_metric,
        args.logs_dir,
        iters=start_iters,
        best_res=best_res,
        grad_clip=args.grad_clip,
        use_cuda=args.cuda,
        loss_weights=loss_weights)

    # Start training
    evaluator.evaluate(
        test_loader,
        step=0,
        tfLogger=eval_tfLogger,
        dataset=test_dataset)
    for epoch in range(start_epoch, args.epochs):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        trainer.train(epoch, train_loader, optimizer, current_lr,
                      print_freq=args.print_freq,
                      train_tfLogger=train_tfLogger,
                      is_debug=args.debug,
                      evaluator=evaluator,
                      test_loader=test_loader,
                      eval_tfLogger=eval_tfLogger,
                      test_dataset=test_dataset)

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset=test_dataset)

    # Close the tensorboard logger
    train_tfLogger.close()
    eval_tfLogger.close()


if __name__ == '__main__':
    # parse the config
    args = get_args(sys.argv[1:])
    main(args)
