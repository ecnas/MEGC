#!/usr/bin/env bash
python train_supernet_emo.py --dataroot database/cityscapes-origin \
  --supernet spade --teacher_netG spade --student_ngf 64 \
  --log_dir logs/gaugan/cityscapes/finetune \
  --tensorboard_dir tensorboards/gaugan/cityscapes/finetune \
  --restore_teacher_G_path logs/gaugan/cityscapes/full/export/latest_net_G.pth \
  --restore_student_G_path logs/gaugan/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/gaugan/cityscapes/supernet/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --load_in_memory --no_fid \
  --nepochs 100 --nepochs_decay 100 \
  --gpu_ids 0,1 \
  --config_str $1
