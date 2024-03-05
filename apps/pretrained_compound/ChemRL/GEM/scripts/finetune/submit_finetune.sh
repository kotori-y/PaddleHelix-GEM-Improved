#!/bin/bash

echo "Choose your task"
select task in "classification" "regression"; do
  break;
done
echo "You have selected $task task"


echo "Choose your finetune model"

select finetune_model in "only_geometry" "geometry_cm5_wiberg" "geometry_espc_wiberg" "geometry_hirshfeld_wiberg" "geometry_npa_wiberg" "cm5_wiberg" "espc_wiberg" "hirshfeld_wiberg" "npa_wiberg" "geometry_allCharges" "no_pretrain"; do
  if [ $finetune_model = "only_geometry" ]; then
    init_model="./pretrained_models/zinc_west/zinc_100/encoder/epoch99.pdparams"
    job="onG"
  elif [ $finetune_model = "geometry_cm5_wiberg" ]; then
    init_model="./pretrained_models/quandb_west/quandb_cm5/encoder/epoch96.pdparams"
    job="cm5"
  elif [ $finetune_model = "geometry_espc_wiberg" ]; then
    init_model="./pretrained_models/quandb_west/quandb_espc/encoder/epoch64.pdparams"
    job="esp"
  elif [ $finetune_model = "geometry_hirshfeld_wiberg" ]; then
    init_model="./pretrained_models/quandb_west/quandb_hirshfeld/encoder/epoch76.pdparams"
    job="hir"
  elif [ $finetune_model = "geometry_npa_wiberg" ]; then
    init_model="./pretrained_models/quandb_west/quandb_npa/encoder/epoch75.pdparams"
    job="npa"
  elif [ $finetune_model = "cm5_wiberg" ]; then
    init_model="./pretrained_models/quandb_west_ex/quandb_cm5/encoder/epoch98.pdparams"
    job="cm5Ex"
  elif [ $finetune_model = "espc_wiberg" ]; then
    init_model="./pretrained_models/quandb_west_ex/quandb_espc/encoder/epoch98.pdparams"
    job="espEx"
  elif [ $finetune_model = "hirshfeld_wiberg" ]; then
    init_model="./pretrained_models/quandb_west_ex/quandb_hirshfeld/encoder/epoch97.pdparams"
    job="hirEx"
  elif [ $finetune_model = "npa_wiberg" ]; then
    init_model="./pretrained_models/quandb_west_ex/quandb_npa/encoder/epoch92.pdparams"
    job="npaEx"
  elif [ $finetune_model = "geometry_allCharges" ]; then
    init_model="./pretrained_models/quandb_west/quandb_all/encoder/epoch60.pdparams"
    job="all"
  else [ $finetune_model = "no_pretrain" ]
    init_model=""
    job="noP"
  fi
  break;
done

sinfo -p gpu,gpu1,gpu3,gpu4,cp1 -O partition:10,nodes:7,nodelist:30,statecompact:10,gres,GresUsed:30sin

echo "Choose your node"

select gpu in "gpu" "gpu1" "gpu3" "gpu4"; do
  break;
done

# echo "yhbatch -N 1 -p $gpu --gpus-per-node=1 -J $job"_"${task^} --cpus-per-gpu=8 -o $finetune_model"_finetune_clsss.log" finetune_class.sh $finetune_model $init_model"

echo "use $init_model"

if [ $task = "classification" ]; then
  yhbatch -N 1 -p $gpu --gpus-per-node=1 -J $job"_"$task --cpus-per-gpu=6 -o $finetune_model"_finetune_clsss.log" finetune_class.sh $finetune_model $init_model
else [ $task = "regression" ]
  yhbatch -N 1 -p $gpu --gpus-per-node=1 -J $job"_"$task --cpus-per-gpu=6 -o $finetune_model"_finetune_regr.log" finetune_regr.sh $finetune_model $init_model
fi
