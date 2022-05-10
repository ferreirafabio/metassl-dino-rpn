#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dinoneps
#SBATCH -t 23:59:59
#SBATCH --array 0-99%10

#SBATCH -o /work/dlclarge2/ferreira-dino/metassl-dino/cluster/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge2/ferreira-dino/metassl-dino/cluster/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID

pip list

source activate dino

python -m main_dino --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/neps_dino_final --batch_size_per_gpu 40 --is_neps_run
