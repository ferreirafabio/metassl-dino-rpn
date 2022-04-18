#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080  # mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:8
#SBATCH -J MSSL_IN
#SBATCH -t 00:59:59

pip list

source activate dino

python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/wagnerd-metassl-experiments/dino/$EXPERIMENT_NAME --batch_size_per_gpu 40 --pretrained_weights /work/dlclarge2/wagnerd-metassl-experiments/dino/$EXPERIMENT_NAME/checkpoint.pth
