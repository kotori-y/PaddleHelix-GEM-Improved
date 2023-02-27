import pickle

import numpy as np
import paddle
from rdkit import Chem
import pgl
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.utils import load_json_config
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_raw3d, mol_to_geognn_graph_data_MMFF3d, \
    mol_to_geognn_graph_data


def load_pickled_mol_to_dataset(data_path: str):
    with open(data_path, 'rb') as f:
        mol_list = pickle.load(f)
    return InMemoryDataset(data_list=mol_list)


def load_sdf_mol_to_dataset(data_path: str):
    mols = Chem.SDMolSupplier(data_path)
    mol_list = [mol for mol in mols]
    return InMemoryDataset(data_list=mol_list)


class ConfGenTaskTransformFn:
    """Gen features for downstream model"""

    def __init__(self, is_inference=False, add_noise=False, only_atom_bond=True):
        self.is_inference = is_inference
        self.add_noise = add_noise
        self.only_atom_bond = only_atom_bond

    def __call__(self, mol):
        gt_pos = mol.GetConformer().GetPositions()
        if self.add_noise:
            noise = np.random.uniform(-1, 1, size=gt_pos.shape)
            gt_pos += noise

        prior_pos = np.random.uniform(-1, 1, gt_pos.shape)
        # prior_pos = gt_pos

        if not self.is_inference:
            data = mol_to_geognn_graph_data(mol, gt_pos - gt_pos.mean(axis=0), dir_type='HT', only_atom_bond=self.only_atom_bond)
        else:
            data = None

        prior_data = mol_to_geognn_graph_data(mol, prior_pos, dir_type='HT', only_atom_bond=self.only_atom_bond)

        return [prior_data, data, prior_pos, gt_pos, mol]


class ConfGenTaskCollateFn(object):
    def __init__(self,
                 atom_names,
                 bond_names,
                 bond_float_names,
                 bond_angle_float_names,
                 dihedral_angle_float_names):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, data_list):
        prior_atom_bond_graph_list = []
        prior_bond_angle_graph_list = []
        prior_angle_dihedral_graph_list = []

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        angle_dihedral_graph_list = []

        batch_list = []
        num_nodes = []
        mol_list = []

        prior_pos_list = []
        gt_pos_list = []

        for i, (prior_data, data, prior_pos, gt_pos, mol) in enumerate(data_list):
            prior_ab_g = pgl.Graph(
                num_nodes=len(prior_data[self.atom_names[0]]),
                edges=prior_data['edges'],
                node_feat={name: prior_data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat={name: prior_data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            prior_ba_g = pgl.Graph(
                num_nodes=len(prior_data['edges']),
                edges=prior_data['BondAngleGraph_edges'],
                node_feat={},
                edge_feat={name: prior_data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            prior_adi_g = pgl.graph.Graph(
                num_nodes=len(prior_data['BondAngleGraph_edges']),
                edges=prior_data['AngleDihedralGraph_edges'],
                node_feat={},
                edge_feat={name: prior_data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names})

            prior_atom_bond_graph_list.append(prior_ab_g)
            prior_bond_angle_graph_list.append(prior_ba_g)
            prior_angle_dihedral_graph_list.append(prior_adi_g)

            if data is not None:
                ab_g = pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
                ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
                adi_g = pgl.graph.Graph(
                    num_nodes=len(data['BondAngleGraph_edges']),
                    edges=data['AngleDihedralGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names})

                atom_bond_graph_list.append(ab_g)
                bond_angle_graph_list.append(ba_g)
                angle_dihedral_graph_list.append(adi_g)

            batch_list.extend([i] * prior_ab_g.num_nodes)
            mol_list.append(mol)
            num_nodes.append(len(mol.GetAtoms()))

            prior_pos_list.append(prior_pos)
            gt_pos_list.append(gt_pos)

        batch_list = np.array(batch_list)
        num_nodes = np.array(num_nodes)
        batch = dict(zip(["batch", "num_nodes", "mols"], [batch_list, num_nodes, mol_list]))

        prior_atom_bond_graph = pgl.Graph.batch(prior_atom_bond_graph_list)
        prior_bond_angle_graph = pgl.Graph.batch(prior_bond_angle_graph_list)
        prior_angle_dihedral_graph = pgl.Graph.batch(prior_angle_dihedral_graph_list)

        if len(atom_bond_graph_list) > 0:
            atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
            bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
            angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)
        else:
            atom_bond_graph = None
            bond_angle_graph = None
            angle_dihedral_graph = None

        # TODO: reshape due to pgl limitations on the shape
        self._flat_shapes(prior_atom_bond_graph.node_feat)
        self._flat_shapes(prior_atom_bond_graph.edge_feat)
        self._flat_shapes(prior_bond_angle_graph.node_feat)
        self._flat_shapes(prior_bond_angle_graph.edge_feat)
        self._flat_shapes(prior_angle_dihedral_graph.node_feat)
        self._flat_shapes(prior_angle_dihedral_graph.edge_feat)

        if atom_bond_graph is not None:
            self._flat_shapes(atom_bond_graph.node_feat)
            self._flat_shapes(atom_bond_graph.edge_feat)
            self._flat_shapes(bond_angle_graph.node_feat)
            self._flat_shapes(bond_angle_graph.edge_feat)
            self._flat_shapes(angle_dihedral_graph.node_feat)
            self._flat_shapes(angle_dihedral_graph.edge_feat)

        return {
            "prior_atom_bond_graphs": prior_atom_bond_graph,
            "prior_bond_angle_graph": prior_bond_angle_graph,
            "prior_angle_dihedral_graph": prior_angle_dihedral_graph,
            "atom_bond_graphs": atom_bond_graph,
            "bond_angle_graph": bond_angle_graph,
            "angle_dihedral_graph": angle_dihedral_graph,
        }, batch, prior_pos_list, gt_pos_list


if __name__ == "__main__":
    _data_path = "./data/qm9/qm9_processed/test_200.sdf"
    dataset = load_sdf_mol_to_dataset(_data_path)
    dataset = dataset[:128]

    transform_fn = ConfGenTaskTransformFn(use_self_pos=True)
    dataset.transform(transform_fn, num_workers=10)

    compound_encoder_config = load_json_config("./data/geognn_l8.json")
    collate_fn = ConfGenTaskCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'],
        bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        dihedral_angle_float_names=compound_encoder_config['dihedral_angle_float_names'])

    train_data_gen = dataset.get_data_loader(
        batch_size=64,
        num_workers=10,
        shuffle=True,
        collate_fn=collate_fn)

    print("DONE!!!")
