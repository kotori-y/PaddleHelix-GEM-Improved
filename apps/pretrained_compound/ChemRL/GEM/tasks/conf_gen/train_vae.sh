#!/bin/bash

# cd ..

source /fs1/software/modules/bashrc
module load loginnode
# module load CUDA/11.2
# module load cudnn/8.4.1-cuda11.x
module load CUDA/10.2
module load cudnn/8.0.5-cuda10.2
export PYTHONUNBUFFERED=1

root_path="../../../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export LD_LIBRARY_PATH='/fs1/home/wangll1/miniconda3/envs/paddlehelix-cu112-gt/lib'

BATCH_SIZE=256
NUM_WORKERS=1
EPOCHS=100
NUM_LAYERS=3

LR=1e-4

DATASET="qm9_bond_angle_diheral_gt_no_position"
LOG_PATH="./log/"

TRAIN_DATA_PATH="./data/raw/train"
VALID_DATA_PATH="./data/raw/valid"

TRAIN_DATA_PATH_DEBUG="./data/raw/test"
VALID_DATA_PATH_DEBUG="./data/raw/test"

PRIOR_CONFIG="./data/configs/prior.json"
ENCODER_CONFIG="./data/configs/encoder.json"
DECODER_CONFIG="./data/configs/decoder.json"
HEAD_CONFIG="./data/configs/head.json"

INIT_MODEL=''

LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-4
DROPOUT_RATE=0.2

N_NOISE_MOL=1
VAE_BETA=1.0

((BATCH_SIZE=$BATCH_SIZE/$N_NOISE_MOL))

DEBUG=$1
DISTRIBUTED=$2

if [ $DEBUG -eq 1 ]; then
  echo "Run as debug mode："
  echo "=================="
  python train_vae.py \
    --batch_size $BATCH_SIZE \
    --num_workers 1 \
    --epochs 10 \
    --num_layers $NUM_LAYERS \
    --dataset 'debug' \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VALID_DATA_PATH_DEBUG \
    --log_path $LOG_PATH \
    --prior_config $PRIOR_CONFIG \
    --encoder_config $ENCODER_CONFIG \
    --decoder_config $DECODER_CONFIG \
    --head_config $HEAD_CONFIG \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --dropout_rate $DROPOUT_RATE \
    --n_noise_mol $N_NOISE_MOL \
    --vae_beta $VAE_BETA \
    --isomorphism \
    --debug
elif [ $DEBUG -eq 0 ] && [ $DISTRIBUTED -eq 0 ]; then
  echo "Run as single gpu mode："
  echo "=================="
  python train_vae.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --num_layers $NUM_LAYERS \
    --dataset $DATASET \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VALID_DATA_PATH \
    --log_path $LOG_PATH \
    --prior_config $PRIOR_CONFIG \
    --encoder_config $ENCODER_CONFIG \
    --decoder_config $DECODER_CONFIG \
    --head_config $HEAD_CONFIG \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --dropout_rate $DROPOUT_RATE \
    --n_noise_mol $N_NOISE_MOL \
    --vae_beta $VAE_BETA \
    --isomorphism
elif [ $DEBUG -eq 0 ] && [ $DISTRIBUTED -eq 1 ]; then
  echo "Run as multi gpu mode："
  echo "=================="
  python -m paddle.distributed.launch train_vae.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs $EPOCHS \
    --num_layers $NUM_LAYERS \
    --dataset $DATASET \
    --train_data_path $TRAIN_DATA_PATH \
    --valid_data_path $VALID_DATA_PATH \
    --log_path $LOG_PATH \
    --prior_config $PRIOR_CONFIG \
    --encoder_config $ENCODER_CONFIG \
    --decoder_config $DECODER_CONFIG \
    --head_config $HEAD_CONFIG \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --dropout_rate $DROPOUT_RATE \
    --n_noise_mol $N_NOISE_MOL \
    --vae_beta $VAE_BETA \
    --distributed
fi
