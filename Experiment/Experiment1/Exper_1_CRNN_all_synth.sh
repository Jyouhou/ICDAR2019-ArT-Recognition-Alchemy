#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/1_CRNN_all_synth \
  --CRNN \
  --train_data_dir "./dataset/synth90k/" \
  --extra_train_dataset synth90k \
  --extra_train_data_dir "./dataset/curvedsynth/"