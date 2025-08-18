#!/bin/bash

#SBATCH --job-name=train_gpt2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zanqil@uw.edu

# #SBATCH --account=sciencehub
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=320G
# #SBATCH --gpus=a40:4
#SBATCH --gpus=l40s:4
#SBATCH --time=01-0:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/gscratch/sciencehub/zq/repro-gpt2
#SBATCH --export=all
source ~/.bashrc
conda activate gpt2
torchrun --standalone --nproc_per_node=4 train_gpt2.py

# sbatch /gscratch/sciencehub/zq/repro-gpt2/train.slurm
