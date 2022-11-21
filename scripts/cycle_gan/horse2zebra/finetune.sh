#!/usr/bin/env bash
python train_supernet_emo.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnet --teacher_netG resnet_9blocks \
  --log_dir logs/cycle_gan/horse2zebra/finetune \
  --gan_mode lsgan \
  --student_ngf 64 --ndf 64 \
  --restore_teacher_G_path logs/cycle_gan/horse2zebra/full/checkpoints/latest_net_G_A.pth  \
  --restore_student_G_path logs/cycle_gan/horse2zebra/supernet/checkpoints/latest_net_G.pth \
  --restore_D_path logs/cycle_gan/horse2zebra/supernet/checkpoints/latest_net_D.pth \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --metaA_path datasets/metas/horse2zebra/train1A.meta \
  --metaB_path datasets/metas/horse2zebra/train1B.meta \
  --config_str $1 \
  --gpu_ids 0
