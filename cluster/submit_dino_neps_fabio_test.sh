#!/bin/bash
#SBATCH -p testdlc_gpu-rtx2080 #mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH -J dino_neps_hpo_test
#SBATCH -t 1:00:00 
##SBATCH --array 0

pip list

conda activate dino
which python

python -m main_dino --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/$EXPERIMENT_NAME --batch_size_per_gpu 40 --is_neps_run --epochs 100


