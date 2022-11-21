#!/usr/bin/env bash
python train_supernet_emo.py --dataroot database/cityscapes-origin \
  --supernet spadeEMO --teacher_netG spade --student_ngf 64 \
  --log_dir logs/gaugan/cityscapes/supernet \
  --restore_teacher_G_path logs/gaugan/cityscapes/full/export/latest_net_G.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --load_in_memory --no_fid \
  --nepochs 200 --nepochs_decay 200 \
  --config_set channels-64 --meta_path datasets/metas/cityscapes-origin/train1.meta \
  --warmup_epochs 0 \
  --evolution_epoch_freq 5 \
  --population_size 20 \
  --gen_num 5 \
  --eval_cnt 1 \
  --gpu_ids 0,1,2,3,4,5,6,7
