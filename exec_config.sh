#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -o LVViT-M_repa_rebuttal_out.txt
#SBATCH -e LVViT-M_repa_rebuttal_err.txt

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_PORT=15771
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)


srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --model lvvit_s --batch-size 128 --img-size 224 --drop-path 0.1 --token-label --token-label-data label_top5_train_nfnet --token-label-size 14 --model-ema --workers=20 --channel_idle --feature_norm=BatchNorm --reparam --amp --native-amp --data_dir /scratch/itee/uqxxu16/data/imagenet

# srun python -m torch.distributed.launch --nproc_per_node=$SLURM_NTASKS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py --model lvvit_m --batch-size 128 --img-size 224 --drop-path 0.1 --token-label --token-label-data label_top5_train_nfnet --token-label-size 14 --model-ema --workers=20 --channel_idle --feature_norm=BatchNorm --reparam --amp --native-amp --data_dir /scratch/itee/uqxxu16/data/imagenet
