#!/usr/bin/env bash
python train_supernet_emo.py --dataroot database/horse2zebra \
  --dataset_mode unaligned \
  --supernet resnetEMO --teacher_netG resnet_9blocks \
  --log_dir logs/cycle_gan/horse2zebra/supernet \
  --gan_mode lsgan \
  --student_ngf 64 --ndf 64 \
  --restore_teacher_G_path logs/cycle_gan/horse2zebra/full/checkpoints/latest_net_G_A.pth  \
  --real_stat_path real_stat/horse2zebra_B.npz \
  --lambda_recon 10 --lambda_distill 0.01 \
  --nepochs 200 --nepochs_decay 200 \
  --save_epoch_freq 20 \
  --config_set channels-64-cycleGAN \
  --metaA_path datasets/metas/horse2zebra/train1A.meta \
  --metaB_path datasets/metas/horse2zebra/train1B.meta \
  --warmup_epochs 300 \
  --evolution_epoch_freq 5 \
  --population_size 20 \
  --gen_num 5 \
  --eval_cnt 1