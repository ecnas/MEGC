#!/usr/bin/env bash
python test.py --dataroot database/cityscapes \
  --results_dir results/pix2pix/cityscapes/compressed \
  --config_str $1 \
  --restore_G_path logs/pix2pix/cityscapes/compressed/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --direction BtoA --need_profile
