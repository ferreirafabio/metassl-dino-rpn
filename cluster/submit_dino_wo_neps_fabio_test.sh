#!/bin/zsh
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_wo_neps_hpo_test
#SBATCH -t 10:00:00
#SBATCH --array 0-10%1

source /home/ferreira/.profile
#source activate dino
source activate dino_neps044
pip show neps

python -m main_dino --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/$EXPERIMENT_NAME --batch_size_per_gpu 40 --epochs 15
