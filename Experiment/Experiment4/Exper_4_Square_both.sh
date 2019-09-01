#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/4_Square_both \
  --train_data_dir "./dataset/synth90k" \
  --extra_train_dataset synthtextlist \
  --extra_train_data_dir \
  "./dataset/curvedsynth" \
  --REC_SQUARE 256 --RotationInTraining=0.15 --REC_ON_INPUT