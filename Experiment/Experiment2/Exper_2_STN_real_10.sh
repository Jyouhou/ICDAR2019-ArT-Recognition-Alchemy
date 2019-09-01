#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/2_STN_real_10 \
  --train_data_dir "./dataset/synth90k" \
  --extra_train_dataset synthtextlist svt totaltext ic03 ic13 ic15 iiit5k ic03  \
  --extra_train_data_dir \
  "./dataset/curvedsynth" \
  "./dataset/svt_train" \
  "./dataset/totaltext_train" \
  "./dataset/ic03_train" \
  "./dataset/ic13_train" \
  "./dataset/ic15_train" \
  "./dataset/iiit5k_train" \
  "./dataset/coco_train" \
  --real_multiplier 69
 
