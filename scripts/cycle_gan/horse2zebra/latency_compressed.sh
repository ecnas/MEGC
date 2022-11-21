#!/usr/bin/env bash
python latency.py --dataroot database/horse2zebra/valA \
  --dataset_mode single \
  --results_dir results/cycle_gan/horse2zebra/compressed \
  --restore_G_path logs/cycle_gan/horse2zebra/compressed/latest_net_G.pth \
  --config_str $1 \
  --need_profile \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --gpu_ids 0
