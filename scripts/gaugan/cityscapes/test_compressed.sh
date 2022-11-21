#!/usr/bin/env bash
python test.py --dataroot database/cityscapes-origin \
  --config_str $1 \
  --model spade --dataset_mode cityscapes \
  --results_dir results/gaugan/cityscapes/compressed \
  --ngf 64 --netG sub_mobile_spade \
  --restore_G_path logs/gaugan/cityscapes/compressed/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt --need_profile \
  --gpu_ids 0
