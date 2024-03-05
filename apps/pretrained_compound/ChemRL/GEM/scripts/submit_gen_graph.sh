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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8

BATCH_SIZE=256
NUM_WORKERS=72

EPOCHS=100
LR=1e-4
DROPOUT=0.2

DATASET='zinc20'

INIT_MODEL='./pretrained_models/zinc20.6/zinc_50/epoch99.pdparams'


TEST_RATIO=0.1

DATASET=zinc_demo
OUT_MODEL='./pretrained_models/zinc20.6/'$DATASET'/epoch99.pdparams'

python pretrain.py \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --max_epoch $EPOCHS \
  --dataset $DATASET \
  --data_path ./data/zinc20.7/zinc_0 \
  --test_ratio $TEST_RATIO \
  --lr $LR \
  --dropout_rate $DROPOUT  \
  --compound_encoder_config ./model_configs/geognn_l8.json \
  --model_config ./model_configs/pretrain_gem.json \
  --model_dir ./pretrained_models/zinc20.6/$DATASET \
  --cached_data_path ./data/cached_data/zinc20.7/zinc_0 \
  --generate_data_only
    # --with_provided_3d
