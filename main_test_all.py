import sys
import argparse
import os

common_cmds = ''
common_cmds += 'CUDA_VISIBLE_DEVICES=0 python3 examples/main.py '
common_cmds += '--workers 4 '
common_cmds += '--evaluate '
common_cmds += '--tps_margins 0.01 0.01 '

parser = argparse.ArgumentParser(
    description='the script to test all datasets.')
parser.add_argument('--logs_dir', type=str, metavar="PATH", default='')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--is_stn', action='store_true', default=False)
parser.add_argument('--CRNN', action='store_true', default=False)
parser.add_argument('--tps_margins', nargs='+', type=float, default=[0, 0])
parser.add_argument('--REC_ON_INPUT', action='store_true', default=False)
parser.add_argument('--tps_outputsize', nargs='+', type=int, default=[64, 256])
parser.add_argument('--ToGrey', action='store_true', default=False)
parser.add_argument(
    '--REC_SQUARE',
    default=0,
    type=int,
    help='')
parser.add_argument(
    '--evaluate_with_lexicon',
    action='store_true',
    default=False)
parser.add_argument('-a', '--arch', type=str, default='ResNet_FPN')

args = parser.parse_args()

common_cmds += '--arch {} '.format(args.arch)

if args.ToGrey:
    common_cmds += "--ToGrey "

if args.logs_dir == '':
    raise ValueError('logs_dir cant empty.')
else:
    if os.path.isfile(args.resume):
        common_cmds += '--logs_dir {0} --resume {1} '.format(
            args.logs_dir + "_test", args.resume)
    else:
        common_cmds += '--logs_dir {0} --resume {1} '.format(
            args.logs_dir+"_test", os.path.join(args.logs_dir, 'model_best.pth.tar'))

if args.evaluate_with_lexicon:
    common_cmds += '--evaluate_with_lexicon '

if args.REC_ON_INPUT:
    common_cmds += '--REC_ON_INPUT --tps_outputsize {0} {1} '.format(
        args.tps_outputsize[0], args.tps_outputsize[1])

if args.CRNN:
    common_cmds += "--CRNN "

if args.REC_SQUARE > 0:
    common_cmds += f"--REC_SQUARE {args.REC_SQUARE} "

common_cmds += '--tps_margins {0} {1} '.format(
    args.tps_margins[0], args.tps_margins[1])

datasets = []
datasets.append(
    ('iiit5k', "===== ===== ===== IIIT5K ===== ===== =====",
     './dataset/iiit5k_test/'))
datasets.append(
    ('svt', "===== ===== ===== SVT ===== ===== =====",
     './dataset/svt_test/'))
datasets.append(
    ('ic03', "===== ===== ===== IC03 ===== ===== =====",
     './dataset/ic03_test/'))
datasets.append(
    ('ic13', "===== ===== ===== IC13 ===== ===== =====",
     './dataset/ic13_test/'))
datasets.append(
    ('ic15', "===== ===== ===== IC15 ===== ===== =====",
     './dataset/ic15_test/'))
datasets.append(
    ('svtp', "===== ===== ===== SVTP ===== ===== =====",
     './dataset/svtp_test/'))
datasets.append(
    ('cute80', "===== ===== ===== CUTE80 ===== ===== =====",
     './dataset/cute80_test/'))
datasets.append(
    ('totaltext', "===== ===== ===== TOTALTEXT ===== ===== =====",
     './dataset/totaltext_test/'))
datasets.append(
    ('ic19_val', "===== ===== ===== IC19_VAL ===== ===== =====",
     './dataset/ic19_val_test/'))
datasets.append(
    ('totaltext', "===== ===== ===== TOTALTEXT-RECTIFIED ===== ===== =====",
     './dataset/RectTotal_test/'))

count = 0
if os.path.isfile(os.path.join(args.logs_dir+'_test', 'log.txt')):
    logfile = os.path.join(args.logs_dir+'_test', 'log.txt')
    content = open(logfile, "r").read()
    for dataset in datasets:
        count += int((f"{dataset[2].split('/')[-1]}" in content) and (f"Finished-{dataset[0]}" in content))

    if count == len(datasets):
        print("test already done!")
        exit(0)

for dataset in datasets:
    cmd = common_cmds + \
        '--test_dataset {0} --test_data_dir {1} '.format(dataset[0], dataset[2])
    if not args.CRNN:
        cmd += '--vis_dir {0}_rectified '.format(dataset[0])
    os.makedirs(f"{args.logs_dir+'_test'}", exist_ok=True)
    logfile = os.path.join(args.logs_dir+'_test', 'log.txt')
    if os.path.isfile(logfile):
        content = open(logfile, "r").read()
        if f"{dataset[2].split('/')[-1]}" in content and f"Finished-{dataset[0]}" in content:
            print(f"{dataset[2].split('/')[-1]} is already tested. skip")
            continue
    cmd += f" >> {os.path.join(args.logs_dir+'_test', 'log.txt')}"
    os.system(cmd)
