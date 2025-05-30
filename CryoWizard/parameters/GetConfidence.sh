#!/bin/bash
#SBATCH -p GPUv100,GPUv100p2,GPUv100p3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --exclusive


