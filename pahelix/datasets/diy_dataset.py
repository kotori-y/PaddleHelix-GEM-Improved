import os
from os.path import join, exists
import pandas as pd
import numpy as np
from rdkit import Chem

from pahelix.datasets.inmemory_dataset import InMemoryDataset


def get_default_diy_task_names():
    """Get that default freesolv task names and return measured expt"""
    return ['endpoint']


def load_diy_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = get_default_diy_task_names()

    # NB: some examples have multiple species
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def get_diy_stat(data_path, task_names):
    """Return mean and std of labels"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }
