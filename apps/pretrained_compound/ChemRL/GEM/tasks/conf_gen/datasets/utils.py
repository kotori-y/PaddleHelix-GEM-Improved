import pickle
from glob import glob
from pathlib import Path

import numpy as np
import paddle
from rdkit import Chem
import pgl
from sklearn.metrics import pairwise_distances

import sys
sys.path.append("../..")
from tasks.conf_gen.utils import scatter_mean, set_rdmol_positions


from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.featurizers.gem_featurizer import get_pretrain_bond_angle, get_pretrain_angle_dihedral
# from pahelix.utils import load_json_config
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_raw3d


def move2origin(poses, batch, num_nodes):
    dim_size = batch.max() + 1
    index = paddle.to_tensor(batch)
    poses_mean = scatter_mean(poses, index, 0, dim_size)
    _poses_mean = poses_mean.numpy().repeat(num_nodes, axis=0)
    _poses_mean = paddle.to_tensor(_poses_mean, dtype=poses_mean.dtype)
    return poses - _poses_mean


def load_pickled_mol_to_dataset(files):
    data_list = []

    for file in files:
        with open(file, 'rb') as f:
            mol_list = pickle.load(f)
        data_list.extend(mol_list)

    return InMemoryDataset(data_list=data_list)


def load_sdf_mol_to_dataset(files):
    """tbd"""
    data_list = []
    for file in files:
        mols = Chem.SDMolSupplier(file)
        for mol in mols:
            data_list.append(mol)
    dataset = InMemoryDataset(data_list=data_list)
    return dataset


def load_mol_to_dataset(data_path: str, debug=False):
    if debug:
        files = sorted(glob('%s/*' % data_path))[:10]
    else:
        files = sorted(glob('%s/*' % data_path))

    if Path(files[0]).suffix == '.pkl':
        return load_pickled_mol_to_dataset(files)
    if Path(files[0]).suffix == '.sdf':
        return load_sdf_mol_to_dataset(files)

    raise ValueError


class ConfGenTaskTransformFn:
    """Gen features for downstream model"""

    def __init__(self, noise=None, n_noise_mol=10, evaluate=False):
        if noise is None:
            noise = [0, 0.5]
        self.n_noise_mol = n_noise_mol
        self.noise = noise
        self.evaluate = evaluate

    def prepare_pretrain_task(self, data, only_atom_bond):
        """
        prepare data for pretrain task
        """
        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])

        if only_atom_bond:
            return data

        node_a, node_b, node_c, node_d, dihedral_angles = \
            get_pretrain_angle_dihedral(
                data["AngleDihedralGraph_edges"], data['BondAngleGraph_edges'],
                data['edges'], data['dihedral_angle'])
        data['Adi_node_a'] = node_a
        data['Adi_node_b'] = node_b
        data['Adi_node_c'] = node_c
        data['Adi_node_d'] = node_d
        data['Adi_angle_dihedral'] = dihedral_angles

        node_i, node_j, node_k, bond_angles = \
            get_pretrain_bond_angle(data['BondAngleGraph_edges'], data['edges'], data['bond_angle'])
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles

        return data

    def __call__(self, mol):
        """
        """
        n_atoms = len(mol.GetAtoms())

        if self.evaluate:
            gt_pos = np.random.uniform(-1, 1, size=(n_atoms, 3))
        else:
            gt_pos = mol.GetConformer().GetPositions()

        prior_pos_list = [np.random.uniform(-1, 1, size=(n_atoms, 3)) for _ in range(self.n_noise_mol)]
        encoder_pos_list = [gt_pos + np.random.uniform(*self.noise, size=(n_atoms, 3)) for _ in range(self.n_noise_mol)]
        decoder_pos = gt_pos
        mol_list = [mol for _ in range(self.n_noise_mol)]
        # decoder_pos = prior_pos_list

        smiles = Chem.MolToSmiles(mol)
        print('smiles', smiles)

        prior_data_list = [
            self.prepare_pretrain_task(
                data=mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, pos), only_atom_bond=False),
                only_atom_bond=False
            )
            for pos in prior_pos_list
        ]

        decoder_data_list = [
            self.prepare_pretrain_task(
                mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, decoder_pos), only_atom_bond=True),
                only_atom_bond=True
            )
            for _ in range(self.n_noise_mol)
        ]

        if self.evaluate:
            return prior_data_list, decoder_data_list

        encoder_data_list = [
            self.prepare_pretrain_task(
                mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, pos), only_atom_bond=False, isomorphic=False),
                only_atom_bond=False
            )
            for pos in encoder_pos_list
        ]

        return prior_data_list, encoder_data_list, decoder_data_list


class ConfGenTaskCollateFn:
    def __init__(
            self,
            atom_names,
            bond_names,
            bond_float_names,
            bond_angle_float_names,
            dihedral_angle_float_names,
            evaluate=False,
            isomorphic=False
         ):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names

        self.evaluate = evaluate
        self.isomorphic = isomorphic

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    def process_data(self, data_list, only_atom_bond=False, prior=False, encoder=False):
        # if pretrain: assert not only_atom_bond

        graph_dict = {}
        feed_dict = {}

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        angle_dihedral_graph_list = []

        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        Ba_bond_angle = []

        Adi_node_a = []
        Adi_node_b = []
        Adi_node_c = []
        Adi_node_d = []
        Adi_angle_dihedral = []

        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []

        batch_list = []
        num_nodes = []
        pos_list = []

        isomorphic = []
        isomorphic_num = []

        node_count = 0
        for i, data in enumerate(data_list):
            position = data['atom_pos']

            N = position.shape[0]

            n_atom = N
            batch_list.extend([i] * n_atom)
            num_nodes.append(n_atom)
            pos_list.append(position)

            if prior:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
            if encoder:
                Bl_bond_length.append(data['Bl_bond_length'])

            ab_g = pgl.Graph(
                num_nodes=N,
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names}
            )

            atom_bond_graph_list.append(ab_g)

            if not only_atom_bond:

                if prior:
                    Ba_node_i.append(data['Ba_node_i'] + node_count)
                    Ba_node_j.append(data['Ba_node_j'] + node_count)
                    Ba_node_k.append(data['Ba_node_k'] + node_count)
                if encoder:
                    Ba_bond_angle.append(data['Ba_bond_angle'])

                if prior:
                    Adi_node_a.append(data['Adi_node_a'] + node_count)
                    Adi_node_b.append(data['Adi_node_b'] + node_count)
                    Adi_node_c.append(data['Adi_node_c'] + node_count)
                    Adi_node_d.append(data['Adi_node_d'] + node_count)
                if encoder:
                    Adi_angle_dihedral.append(data['Adi_angle_dihedral'])

                E = len(data['edges'])
                A = len(data['BondAngleGraph_edges'])

                ba_g = pgl.Graph(
                    num_nodes=E,
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names}
                )

                adi_g = pgl.graph.Graph(
                    num_nodes=A,
                    edges=data['AngleDihedralGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names}
                )

                bond_angle_graph_list.append(ba_g)
                angle_dihedral_graph_list.append(adi_g)

                if encoder and self.isomorphic:
                    isomorphic += [*data['isomorphic']]
                    isomorphic_num.append(len(data['isomorphic']))

        if prior:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
        if encoder:
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')

        if prior:
            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
        if encoder:
            feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')

        if prior:
            feed_dict['Adi_node_a'] = np.concatenate(Adi_node_a, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_b'] = np.concatenate(Adi_node_b, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_c'] = np.concatenate(Adi_node_c, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_d'] = np.concatenate(Adi_node_c, 0).reshape(-1).astype('int64')
        if encoder:
            feed_dict['Adi_angle_dihedral'] = np.concatenate(Adi_angle_dihedral, 0).reshape(-1, 1).astype('float32')

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        graph_dict["atom_bond_graph"] = atom_bond_graph

        if not only_atom_bond:
            bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
            self._flat_shapes(bond_angle_graph.node_feat)
            self._flat_shapes(bond_angle_graph.edge_feat)
            graph_dict["bond_angle_graph"] = bond_angle_graph

            angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)
            self._flat_shapes(angle_dihedral_graph.node_feat)
            self._flat_shapes(angle_dihedral_graph.edge_feat)
            graph_dict['angle_dihedral_graph'] = angle_dihedral_graph

        batch_list = np.array(batch_list)
        num_nodes = np.array(num_nodes)
        pos_list = np.vstack(pos_list)

        if encoder and self.isomorphic:
            batch = {
                "batch": batch_list,
                "num_nodes": num_nodes,
                "positions": pos_list,
                "isomorphic": np.hstack(isomorphic),
                "isomorphic_num": np.array(isomorphic_num)
            }
            return graph_dict, feed_dict, batch

        batch = {
            "batch": batch_list,
            "num_nodes": num_nodes,
            "positions": pos_list
        }
        return graph_dict, feed_dict, batch

    def __call__(self, batch_data_list):
        prior_data_list = [x[0] for x in batch_data_list]
        decoder_data_list = [x[-1] for x in batch_data_list]

        prior_data_list = sum(prior_data_list, [])
        decoder_data_list = sum(decoder_data_list, [])

        prior_graph, prior_feed, prior_batch = self.process_data(
            prior_data_list, only_atom_bond=False, prior=True, encoder=False
        )

        decoder_graph, decoder_feed, decoder_batch = self.process_data(
            decoder_data_list, only_atom_bond=True, prior=False, encoder=False
        )

        if self.evaluate:
            return prior_graph, decoder_graph, \
                prior_feed, decoder_feed, \
                prior_batch, decoder_batch

        if not self.evaluate:
            encoder_data_list = [x[1] for x in batch_data_list]
            encoder_data_list = sum(encoder_data_list, [])

            encoder_graph, encoder_feed, encoder_batch = self.process_data(
                encoder_data_list, only_atom_bond=False, prior=False, encoder=True
            )

            return prior_graph, encoder_graph, decoder_graph,\
                prior_feed, encoder_feed, decoder_feed,\
                prior_batch, encoder_batch, decoder_batch
