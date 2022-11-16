#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_rpn_s_symmetric_bounded_no_warmup
#SBATCH -t 23:59:59 #4-23:59:59 #23:59:59
#SBATCH --array 0-30%1


source /home/ferreira/.profile
source activate dinorpn

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino-rpn/metassl-dino-rpn/experiments/$EXPERIMENT_NAME --batch_size_per_gpu $BATCH_SIZE --local_crops_number 2 --saveckp_freq 10 --epochs $EPOCHS --warmup_epochs $WARMUP_EPOCHS --use_rpn_optimizer $USE_RPN_OPTIMIZER --invert_rpn_gradients $INVERT_GRADIENTS --separate_localization_net $SEPARATE_LOCAL_NET --stn_mode $STN_MODE --use_fp16 True --use_bn True --rpn_warmup_epochs 0 --use_unbounded_stn False
