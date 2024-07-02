#!/bin/bash
#SBATCH -J point2       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 5          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04
#SBATCH --qos=high

python train_seg.py