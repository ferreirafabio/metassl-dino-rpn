#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 # partition (queue)
#SBATCH -t 0-23:59 # time (D-HH:MM)
#SBATCH --gres=gpu:8

#SBATCH -o /work/dlclarge2/ferreira-dino/metassl-dino/cluster/log/%x.%N.%A.%a.out # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e /work/dlclarge2/ferreira-dino/metassl-dino/cluster/log/%x.%N.%A.%a.err # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --job-name test_ddp_dino # sets the job name. If not specified, the file name will be used as job name

export HTTP_PROXY=http://tfproxy.informatik.uni-freiburg.de:8080
export HTTPS_PROXY=https://tfproxy.informatik.uni-freiburg.de:8080
export http_proxy=http://tfproxy.informatik.uni-freiburg.de:8080
export https_proxy=https://tfproxy.informatik.uni-freiburg.de:8080

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

pwd

echo "source activate"
source activate dino
echo "run script"
export PYTHONPATH=$PYTHONPATH:.
python main_dino.py --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/test_ddp_dino --batch_size_per_gpu 40 --is_neps_run --epochs 1
echo "done"
