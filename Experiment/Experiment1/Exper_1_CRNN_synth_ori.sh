#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/1_CRNN_synth_ori \
  --CRNN \
  --train_dataset synthtextlist \
  --train_data_dir "./dataset/synthtext"
