#!/bin/zsh
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_neps_full_train_dataset
#SBATCH -t 5-23:59:59  # 23:59:59
#SBATCH --output=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/config_8_2_full_train_dataset/%x.%A.%a.%N.err_out 
#SBATCH --error=/work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/config_8_2_full_train_dataset/%x.%A.%a.%N.err_out
##SBATCH --array 0-9999%1

#source /home/ferreira/.zshrc
source /home/ferreira/.profile
source activate dino
pip show neps

python -m main_dino --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/config_8_2_full_train_dataset --batch_size_per_gpu 40 --epochs 100 --lr 0.0009012 --out_dim 74444 --momentum_teacher 0.9917698 --warmup_teacher_temp 0.0522498 --warmup_teacher_temp_epochs 5 --weight_decay 0.0702967 --weight_decay_end 0.3754394 --freeze_last_layer 2 --warmup_epochs 18 --min_lr 0.0000011 --drop_path_rate 0.0800475 --optimizer 'adamw' --use_bn_in_head False --norm_last_layer True
