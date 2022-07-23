#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_wo_neps_pretraining
#SBATCH -t 23:59:59 #5-23:59:59  # 23:59:59
#SBATCH --array 0-10%1

#source /home/ferreira/.profile
#source activate dino_neps044
#pip show neps

source /home/ferreira/.profile
source activate dino

python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes=1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME --batch_size_per_gpu 40 --world_size 8 --gpu 8

#python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes=1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/$EXPERIMENT_NAME --batch_size_per_gpu 40 --epochs 5 --world_size 8 --gpu 8 --warmup_teacher_temp_epochs 0 --warmup_epochs 1
