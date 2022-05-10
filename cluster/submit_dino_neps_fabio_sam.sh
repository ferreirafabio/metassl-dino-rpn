#!/bin/zsh
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_neps_hpo
#SBATCH -t 5-23:59:59  # 23:59:59
#SBATCH --array 0-9999%10

source /home/ferreira/.profile
source activate dino

mkdir -p /tmp/dino_communication

filename=/tmp/dino_communication/$(openssl rand -hex 12)

python -u -m torch.distributed.launch --use_env --nproc_per_node=8 --nnodes=1 main_dino.py --config_file_path $filename --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/$EXPERIMENT_NAME --batch_size_per_gpu 40 --is_neps_run --epochs 100 --world_size 8 --gpu 8

rm filename

