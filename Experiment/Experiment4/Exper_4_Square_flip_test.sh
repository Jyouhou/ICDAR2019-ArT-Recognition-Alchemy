#!/usr/bin/env bash
python3 main_test_all.py \
  --logs_dir ./logs/4_Square_flip  --args.resume=./pretrained_models/Squarization_random_rotation.path.tar \
  --REC_SQUARE 256
