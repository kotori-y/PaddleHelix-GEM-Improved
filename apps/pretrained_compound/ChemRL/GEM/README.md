# Quantum-enhanced Molecular 3D Representation Learning for Property Prediction

# Background
*TODO*


# Installation guide
## Prerequisites

* OS support: Linux
* Python version: 3.6, 3.7, 3.8

## Dependencies

| name         | version |
| ------------ | ---- |
| numpy        | - |
| pandas       | - |
| networkx     | - |
| paddlepaddle | \>=2.0.0 |
| pgl          | \>=2.1.5 |
| rdkit-pypi   | - |
| sklearn      | - |

('-' means no specific version requirement for that package)

# Usage

Firstly, download or clone the lastest github repository:

```shell
git clone https://github.com/kotori-y/PaddleHelix-GEM-Improved.git
cd PaddleHelix-GEM-Improved
git checkout gem-advanced
cd apps/pretrained_compound/ChemRL/GEM
```

## Pretraining
Use the following command to download the demo data which is a tiny subset [Zinc Dataset](https://zinc.docking.org/) and run pretrain tasks.

```shell
cd scripts/pretrain && sh pretrain_ddp.sh
```

The pretrained model will be save under `./pretrain_models`.

We also provide a series pretrained models [here](https://github.com/kotori-y/SomeData/releases/download/v1.0/pretrained_models.tar.gz) for reproducing the downstream finetuning results. Also, the pretrained model can be used for other molecular property prediction tasks.

## Downstream finetuning
After the pretraining, the downstream tasks can use the pretrained model as initialization. 

Firstly, download the pretrained model from the previous step:

```shell
# cd PaddleHelix-GEM-Improved/apps/pretrained_compound/ChemRL/GEM
wget https://github.com/kotori-y/SomeData/releases/download/v1.0/pretrained_models.tar.gz
tar -zxvf pretrain_models-chemrl_gem.tgz
```

Download the downstream molecular property prediction datasets from [MoleculeNet](http://moleculenet.ai/), including classification tasks and regression tasks:

```shell
# cd PaddleHelix-GEM-Improved/apps/pretrained_compound/ChemRL/GEM
wget https://github.com/kotori-y/SomeData/releases/download/v1.0/chemrl_downstream_datasets.tar.gz
tar -zxvf chemrl_downstream_datasets.tgz
```

Run downstream finetuning and the final results will be saved under `./log/pretrain-$dataset/final_result`. 

```shell
cd scripts/finetune && sh submit_finetune.sh
```
