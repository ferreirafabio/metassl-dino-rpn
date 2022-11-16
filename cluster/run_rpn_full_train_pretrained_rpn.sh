#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_rpn_pretrained_rpn
#SBATCH -t 23:59:59
#SBATCH --array 0-25%1

source /home/ferreira/.profile
source activate dinorpn
#source activate dino_newpt
#source activate dino2 
#cuda11.8

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/$EXPERIMENT_NAME --batch_size_per_gpu $BATCH_SIZE --local_crops_number 2 --saveckp_freq 10 --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS --separate_localization_net $SEPARATE_LOCAL_NET --rpn_pretrained_weights $RPN_PRETRAINED_WEIGHTS --stn_mode $STN_MODE
