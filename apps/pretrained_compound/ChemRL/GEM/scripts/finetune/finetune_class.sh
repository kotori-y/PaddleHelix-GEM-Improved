#!/bin/bash

cd ../..

# source ~/.bashrc
source ./scripts/utils.sh

source /fs1/software/modules/bashrc
module load loginnode
module load CUDA/11.2
module load cudnn/8.4.1-cuda11.x
export PYTHONUNBUFFERED=1

# echo "Choose your finetune model"

# select finetune_model in "only_geometry" "geometry_cm5_wiberg" "geometry_espc_wiberg" "geometry_espc_wiberg" "geometry_hirshfeld_wiberg" "geometry_npa_wiberg" "allCharges" "no_pretrain"; do
finetune_model=$1
init_model=$2

echo "[classification] You have selected $finetune_model with $init_model"

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
# datasets="bace bbbp clintox sider tox21 toxcast hiv muv"
datasets="muv toxcast"
compound_encoder_config="model_configs/geognn_l8.json"

# init_model="./pretrained_models/zinc_west/zinc_100/encoder/epoch99.pdparams"
# init_model=""
# init_model="./pretrained_models/quandb_west/quandb_npa/model/epoch75.pdparams"

my_prefix=$finetune_model
log_prefix="log_west/finetune/$finetune_model/$finetune_model"
thread_num=4
count=0
for dataset in $datasets; do
	echo "==> $dataset"
	data_path="./data/chemrl_downstream_datasets/$dataset"
	cached_data_path="./data/cached_data/$dataset"
	if [ ! -f "$cached_data_path.done" ]; then
		rm -r $cached_data_path
		 yhrun -N 1 -p gpu1 --gpus-per-node=1 --cpus-per-gpu=4 python finetune_class.py \
				--task=data \
				--num_workers=10 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--compound_encoder_config=$compound_encoder_config \
				--model_config="model_configs/down_mlp2.json"
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
		touch $cached_data_path.done
	fi

	model_config_list="model_configs/down_mlp2.json model_configs/down_mlp3.json"
	# lrs_list="1e-3,1e-3 4e-3,4e-3 1e-4,1e-3 1e-4,1e-4"
	lrs_list="1e-3,1e-3 4e-3,4e-3 1e-4,1e-3 1e-4,1e-4 1e-4,4e-3"
	drop_list="0.0 0.1 0.2 0.3 0.4"
	# drop_list="0.0 0.4"
	if [ "$dataset" == "tox21" ] || [ "$dataset" == "toxcast" ]; then
		batch_size=128
	else
		batch_size=128 # 32
	fi
	for model_config in $model_config_list; do
		for lrs in $lrs_list; do
			IFS=, read -r -a array <<< "$lrs"
			lr=${array[0]}
			head_lr=${array[1]}
			for dropout_rate in $drop_list; do
				log_dir="$log_prefix-$dataset"
				log_file="$log_dir/lr${lr}_${head_lr}-drop${dropout_rate}.txt"
				echo "Outputs redirected to $log_file"
				mkdir -p $log_dir
				for time in $(seq 1 4); do
					{
						CUDA_VISIBLE_DEVICES=0 python finetune_class.py \
								--batch_size=$batch_size \
								--max_epoch=100 \
								--dataset_name=$dataset \
								--data_path=$data_path \
								--cached_data_path=$cached_data_path \
								--split_type=scaffold \
								--compound_encoder_config=$compound_encoder_config \
								--model_config=$model_config \
								--init_model=$init_model \
								--model_dir=./finetune_models/$dataset \
								--encoder_lr=$lr \
								--head_lr=$head_lr \
								--dropout_rate=$dropout_rate >> $log_file 2>&1
						cat $log_dir/* | grep FINAL| python ana_results.py > $log_dir/final_result
					} &
					let count+=1
					if [[ $(($count % $thread_num)) -eq 0 ]]; then
						wait
					fi
				done
			done
		done
	done
done
wait

