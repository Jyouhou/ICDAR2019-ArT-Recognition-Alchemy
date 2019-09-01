#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/1_STN_all_synth \
  --train_data_dir "./dataset/synth90k/" \
  --extra_train_dataset synth90k \
  --extra_train_data_dir "./dataset/curvedsynth/"