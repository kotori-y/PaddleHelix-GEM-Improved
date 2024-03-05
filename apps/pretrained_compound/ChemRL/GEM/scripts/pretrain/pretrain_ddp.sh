#!/bin/bash

cd ../..

source /fs1/software/modules/bashrc
module load loginnode
module load CUDA/11.2
module load cudnn/8.4.1-cuda11.x
export PYTHONUNBUFFERED=1

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

BATCH_SIZE=128
NUM_WORKERS=72

EPOCHS=100
LR=1e-4
DROPOUT=0.2

# DATASET='zinc20'

INIT_MODEL=''

TEST_RATIO=0.1

# DATASET=zinc_$i
# OUT_MODEL='./pretrained_models/zinc20.5/'$DATASET'/epoch99.pdparams'

# nvidia-smi -l 5 2>&1 >gpu_info.log & #检测主节点gpu使用情况

GPU_NODE='gpu3'
GPU_NUM=2

DATA_ROOT='zinc_west'

for i in {1..100}
do
    DATASET=zinc_$i
    OUT_MODEL='./pretrained_models/'$DATA_ROOT'/'$DATASET'/model/epoch99.pdparams'
    if [ $i -eq 1 ]; then
        python -m paddle.distributed.launch pretrain.py \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --max_epoch $EPOCHS \
        --dataset $DATASET@$DATA_ROOT \
        --data_path ./data/$DATA_ROOT/$DATASET \
        --test_ratio $TEST_RATIO \
        --lr $LR \
        --dropout_rate $DROPOUT  \
        --compound_encoder_config ./model_configs/geognn_l8.json \
        --model_config ./model_configs/pretrain_gem.json \
        --model_dir ./pretrained_models/$DATA_ROOT/$DATASET \
	--distributed
        # --with_provided_3d
    else
        python -m paddle.distributed.launch pretrain.py \
          --batch_size $BATCH_SIZE \
          --num_workers $NUM_WORKERS \
          --max_epoch $EPOCHS \
          --dataset $DATASET@$DATA_ROOT \
          --data_path ./data/$DATA_ROOT/$DATASET \
          --test_ratio $TEST_RATIO \
          --lr $LR \
          --dropout_rate $DROPOUT  \
          --compound_encoder_config ./model_configs/geognn_l8.json \
          --model_config ./model_configs/pretrain_gem.json \
          --model_dir ./pretrained_models/$DATA_ROOT/$DATASET \
          --init_model=$INIT_MODEL \
	  --distributed
          # --with_provided_3d
    fi
    INIT_MODEL=$OUT_MODEL
    
done

