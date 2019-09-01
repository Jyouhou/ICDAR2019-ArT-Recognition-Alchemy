#!/usr/bin/env bash
python3 examples/main.py \
  --logs_dir ./logs/6_ic19_152 \
  --train_data_dir "./dataset/synth90k" \
  --extra_train_dataset synthtextlist iiit5k svt ic03 ic13 ic15 svtp cute80 svt ic03 ic13 ic15 iiit5k ic03 ic19 \
  --extra_train_data_dir \
  "./dataset/curvedsynth" \
  "./dataset/iiit5k_test" \
  "./dataset/svt_test" \
  "./dataset/ic03_test" \
  "./dataset/ic13_test" \
  "./dataset/ic15_test" \
  "./dataset/svtp_test" \
  "./dataset/cute80_test" \
  "./dataset/svt_train" \
  "./dataset/ic03_train" \
  "./dataset/ic13_train" \
  "./dataset/ic15_train" \
  "./dataset/iiit5k_train" \
  "./dataset/coco_test" \
  "./dataset/ic19_train" \
  --real_multiplier 46 --num_layers=152  --voc_type ALLCASES_SYMBOLS
