#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/1_CRNN_synth_90k \
  --CRNN \
  --train_data_dir "./dataset/synth90k" --ToGrey 