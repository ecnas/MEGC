#!/usr/bin/env bash
python train_supernet_emo.py --dataroot database/cityscapes \
  --supernet resnet \
  --log_dir logs/pix2pix/cityscapes/finetune \
  --real_stat_path real_stat/cityscapes_A.npz \
  --restore_teacher_G_path logs/pix2pix/cityscapes/full/checkpoints/latest_net_G.pth \
  --restore_student_G_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_D.pth \
  --config_str $1 \
  --nepochs 200 --nepochs_decay 300 \
  --save_latest_freq 25000 --save_epoch_freq 25 \
  --teacher_ngf 64 --student_ngf 64 --teacher_netG resnet_9blocks \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/val_table.txt \
  --eval_batch_size 2 \
  --direction BtoA --meta_path datasets/metas/cityscapes/train1.meta \
  --gpu_ids 0
