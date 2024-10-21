import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from pahelix.datasets.inmemory_dataset import InMemoryDataset

__all__ = ['load_localprop_dataset']


def load_localprop_dataset(data_path, task_name, debug=False):
    data_list = []

    raw_path = os.path.join(data_path, 'raw/prop_sdf')
    sdf_files = os.listdir(raw_path)

    if debug:
        sdf_files = sdf_files[:100]

    for sdf_file in sdf_files:
        data = {}
        mol = Chem.SDMolSupplier(os.path.join(raw_path, sdf_file))[0]
        if mol is None:
            continue
        data['mol'] = mol
        data['smiles'] = Chem.MolToSmiles(mol)
        data['label'] = np.array(json.loads(mol.GetProp(task_name))).astype('float32')
        data_list.append(data)

    return InMemoryDataset(data_list)
