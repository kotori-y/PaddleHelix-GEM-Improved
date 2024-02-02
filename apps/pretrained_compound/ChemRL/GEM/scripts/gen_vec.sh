#!/bin/bash

cd ..

#source ~/.bashrc
#source ./scripts/utils.sh

#source /GPUFS/paratera/scripts/cn-module.sh
#module load CUDA/10.2 compiler/CUDA/10.2
#module load anaconda/2020.11
#module load nccl/2.9.6-1_cuda10.2
#module load cudnn/7.6.5.32_cuda10.2
module load cuda/11.3
source activate paddlehelix
export PYTHONUNBUFFERED=1

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=8

CUDA_VISIBLE_DEVICES=8 python gen_vec.py --dataset_path ./data/mol_with_3d --compound_encoder_config ./model_configs/geognn_l8.json --with_provided_3d
