#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/3_Square \
  --train_data_dir "./dataset/synth90k" \
  --extra_train_dataset synthtextlist \
  --extra_train_data_dir \
  "./dataset/curvedsynth" \
  --REC_SQUARE 256 
