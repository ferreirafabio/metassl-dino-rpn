#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 #mldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J DINO_RPN_EVAL
#SBATCH -t 23:59:59
#SBATCH --array 0-5%1


source /home/ferreira/.profile
#source activate dino
#source activate dino_newpt
source activate dinorpn

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 eval_linear.py --data_path /data/datasets/ImageNet/imagenet-pytorch --output_dir /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/$EXPERIMENT_NAME --pretrained_weights /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/$EXPERIMENT_NAME/checkpoint.pth --seed $SEED --dataset ImageNet --batch_size_per_gpu 16
