import argparse
from glob import glob

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
from rdkit import Chem

from apps.pretrained_compound.ChemRL.GEM.src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from apps.pretrained_compound.ChemRL.GEM.src.utils import get_dataset, get_downstream_task_names
from pahelix.datasets import InMemoryDataset
from pahelix.model_zoo.gem_model import GeoGNNModel, GeoGNNModelOld
from pahelix.utils import load_json_config


def load_user_dataset(data_path, smiles_col):
    input_df = pd.read_csv(data_path, sep=',')
    smiles_list = input_df[smiles_col]
    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]
        data_list.append(data)
    dataset = InMemoryDataset(data_list)
    return dataset


def load_3dmol_to_dataset(data_path):
    """tbd"""
    files = sorted(glob('%s/*' % data_path))
    data_list = []
    for file in files:
        mols = Chem.SDMolSupplier(file)
        data_list.extend([mol for mol in mols])
    dataset = InMemoryDataset(data_list=data_list)
    return dataset


class VectorGeneratorModel(nn.Layer):
    def __init__(self, compound_encoder, use_old=False):
        super(VectorGeneratorModel, self).__init__()
        self.compound_encoder = compound_encoder
        self.use_old = use_old
        # self.norm = nn.LayerNorm(compound_encoder.graph_dim)

    def forward(self, atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs):
        if not self.use_old:
            _, _, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs)
        else:
            _, _, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs)
        return graph_repr


@paddle.no_grad()
def generate(args, model: VectorGeneratorModel, dataset, collate_fn):
    """
    """
    data_gen = dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    total_vec = []

    model.eval()
    for atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs in data_gen:
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        angle_dihedral_graphs = angle_dihedral_graphs.tensor()

        vec = model(atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs)
        total_vec.append(vec)

    total_vec = np.concatenate(total_vec, 0)
    print(total_vec.shape)

    return total_vec


def main(args):
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    compound_encoder_config['dropout_rate'] = 0.0

    if not args.use_old:
        compound_encoder = GeoGNNModel(compound_encoder_config)
    else:
        compound_encoder = GeoGNNModelOld(compound_encoder_config)
    model = VectorGeneratorModel(compound_encoder, use_old=args.use_oldshu)
    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    if not args.with_provided_3d:
        dataset = load_user_dataset(args.dataset_path, args.smiles_col)
    else:
        dataset = load_3dmol_to_dataset(args.dataset_path)

    dataset.transform(
        DownstreamTransformFn(
            is_inference=True, with_provided_3d=args.with_provided_3d, shuffle_coord=args.shuffle_coord
        ), num_workers=args.num_workers
    )

    collate_fn = DownstreamCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        dihedral_angle_float_names=compound_encoder_config['dihedral_angle_float_names'],
        task_type='na',
        is_inference=True
    )

    vector = generate(args, model, dataset, collate_fn)
    np.save(f'{args.out_dir}', vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_provided_3d", action='store_true', default=False)
    parser.add_argument("--shuffle_coord", action='store_true', default=False)
    parser.add_argument("--use_old", iaction='store_true', default=False)

    parser.add_argument("--num_workers", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default='')
    parser.add_argument("--compound_encoder_config", type=str)

    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()
    print(args)

    main(args)
