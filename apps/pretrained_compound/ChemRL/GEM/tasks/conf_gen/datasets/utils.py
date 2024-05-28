import hashlib
import pickle
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import paddle
import pgl
from rdkit import Chem

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

    def __init__(self, noise=None, n_noise_mol=10, evaluate=False, isomorphism=False):
        if noise is None:
            noise = [0, 0.5]
        self.n_noise_mol = n_noise_mol
        self.noise = noise
        self.evaluate = evaluate
        self.isomorphism = isomorphism

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

        # ring_edges = data['edges'][data['is_in_ring'] != 1]

        mask = (node_a == node_b) | (node_a == node_c) | (node_a == node_d) | \
               (node_b == node_c) | (node_b == node_d) | (node_c == node_d)

        # med_edges = np.stack([node_b, node_c], axis=1)
        # ring_mask = np.array(list(map(lambda x: (ring_edges == x).all(axis=1).any(), med_edges)))

        # mask |= ring_mask

        data['Adi_node_a'] = node_a[~mask]
        data['Adi_node_b'] = node_b[~mask]
        data['Adi_node_c'] = node_c[~mask]
        data['Adi_node_d'] = node_d[~mask]
        data['Adi_angle_dihedral'] = np.array(dihedral_angles)[~mask]

        node_i, node_j, node_k, bond_angles = \
            get_pretrain_bond_angle(data['BondAngleGraph_edges'], data['edges'], data['bond_angle'])

        mask = (node_i == node_j) | (node_i == node_k) | (node_j == node_k)

        data['Ba_node_i'] = node_i[~mask]
        data['Ba_node_j'] = node_j[~mask]
        data['Ba_node_k'] = node_k[~mask]
        data['Ba_bond_angle'] = np.array(bond_angles)[~mask]

        return data

    def __call__(self, mol):
        """
        """
        n_atoms = len(mol.GetAtoms())

        if self.evaluate:
            gt_pos = np.random.uniform(-1, 1, size=(n_atoms, 3))
        else:
            gt_pos = mol.GetConformer().GetPositions()

        # prior_pos_list = [np.random.uniform(-1, 1, size=(n_atoms, 3)) for _ in range(self.n_noise_mol)]
        prior_pos_list = [gt_pos for _ in range(self.n_noise_mol)]
        # encoder_pos_list = [gt_pos + np.random.uniform(*self.noise, size=(n_atoms, 3)) for _ in range(self.n_noise_mol)]
        encoder_pos_list = [gt_pos for _ in range(self.n_noise_mol)]
        decoder_pos = gt_pos
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
                mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, decoder_pos), only_atom_bond=False),
                only_atom_bond=False
            )
            for _ in range(self.n_noise_mol)
        ]

        if self.evaluate:
            return prior_data_list, decoder_data_list

        encoder_data_list = [
            self.prepare_pretrain_task(
                mol_to_geognn_graph_data_raw3d(
                    set_rdmol_positions(mol, pos), only_atom_bond=False, isomorphism=self.isomorphism
                ),
                only_atom_bond=False
            )
            for pos in encoder_pos_list
        ]

        mol_list = [mol for _ in range(self.n_noise_mol)]

        return prior_data_list, encoder_data_list, decoder_data_list, mol_list


class ConfGenTaskCollateFn:
    def __init__(
            self,
            atom_names,
            bond_names,
            bond_float_names,
            bond_angle_float_names,
            dihedral_angle_float_names,
            evaluate=False,
            isomorphism=False
    ):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names

        self.evaluate = evaluate
        self.isomorphism = isomorphism

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    @staticmethod
    def generate_mask(data, masked_atoms):
        Bl_node_in_masked_atoms = (np.isin(data['Bl_node_i'], masked_atoms)) | \
                                  (np.isin(data['Bl_node_j'], masked_atoms))

        Ba_node_in_masked_atoms = (np.isin(data['Ba_node_i'], masked_atoms)) | \
                                  (np.isin(data['Ba_node_j'], masked_atoms)) | \
                                  (np.isin(data['Ba_node_k'], masked_atoms))

        Adi_node_in_masked_atoms = (np.isin(data['Adi_node_a'], masked_atoms)) | \
                                   (np.isin(data['Adi_node_b'], masked_atoms)) | \
                                   (np.isin(data['Adi_node_c'], masked_atoms)) | \
                                   (np.isin(data['Adi_node_d'], masked_atoms))

        return Bl_node_in_masked_atoms, Ba_node_in_masked_atoms, Adi_node_in_masked_atoms

    def process_data(self, data_list, prior=False, encoder=False, masked_nodes=None):
        # if pretrain: assert not only_atom_bond

        graph_dict = {}
        feed_dict = {}

        if masked_nodes is None:
            masked_nodes = []

        # atom_bond_graph_list = []
        # bond_angle_graph_list = []
        # angle_dihedral_graph_list = []

        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_angle_dihedral_graph_list = []

        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        # Ba_bond_angle = []
        #
        Adi_node_a = []
        Adi_node_b = []
        Adi_node_c = []
        Adi_node_d = []
        # Adi_angle_dihedral = []
        #
        Bl_node_i = []
        Bl_node_j = []
        # Bl_bond_length = []

        masked_Ba_node_i = []
        masked_Ba_node_j = []
        masked_Ba_node_k = []
        masked_Ba_bond_angle = []

        masked_Adi_node_a = []
        masked_Adi_node_b = []
        masked_Adi_node_c = []
        masked_Adi_node_d = []
        masked_Adi_angle_dihedral = []

        masked_Bl_node_i = []
        masked_Bl_node_j = []
        masked_Bl_bond_length = []

        batch_list = []
        num_nodes = []
        pos_list = []

        isomorphism = []
        isomorphism_num = []

        node_count = 0
        for i, data in enumerate(data_list):
            position = data['atom_pos']

            N = position.shape[0]

            n_atom = N
            batch_list.extend([i] * n_atom)
            num_nodes.append(n_atom)
            pos_list.append(position)

            ab_g = pgl.Graph(
                num_nodes=N,
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names}
            )

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

            if masked_nodes is None or len(masked_nodes) <= i:
                masked_ab_g, masked_ba_g, masked_adi_g, mask_node_i, _ = \
                    mask_context_of_geognn_graph(ab_g, ba_g, adi_g, mask_ratio=0.15)
                masked_nodes.append(mask_node_i)
            else:
                masked_ab_g, masked_ba_g, masked_adi_g, mask_node_i, _ = \
                    mask_context_of_geognn_graph(ab_g, ba_g, adi_g, mask_ratio=0.15, target_atom_indices=masked_nodes[i])
            # atom_bond_graph_list.append(ab_g)
            # bond_angle_graph_list.append(ba_g)
            # angle_dihedral_graph_list.append(adi_g)

            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_angle_dihedral_graph_list.append(masked_adi_g)

            if prior:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)

                Ba_node_i.append(data['Ba_node_i'] + node_count)
                Ba_node_j.append(data['Ba_node_j'] + node_count)
                Ba_node_k.append(data['Ba_node_k'] + node_count)

                Adi_node_a.append(data['Adi_node_a'] + node_count)
                Adi_node_b.append(data['Adi_node_b'] + node_count)
                Adi_node_c.append(data['Adi_node_c'] + node_count)
                Adi_node_d.append(data['Adi_node_d'] + node_count)

                Bl_node_in_masked_atoms, Ba_node_in_masked_atoms, Adi_node_in_masked_atoms = \
                    self.generate_mask(data, mask_node_i)

                masked_Bl_node_i.append(data['Bl_node_i'][Bl_node_in_masked_atoms] + node_count)
                masked_Bl_node_j.append(data['Bl_node_j'][Bl_node_in_masked_atoms] + node_count)

                masked_Ba_node_i.append(data['Ba_node_i'][Ba_node_in_masked_atoms] + node_count)
                masked_Ba_node_j.append(data['Ba_node_j'][Ba_node_in_masked_atoms] + node_count)
                masked_Ba_node_k.append(data['Ba_node_k'][Ba_node_in_masked_atoms] + node_count)

                masked_Adi_node_a.append(data['Adi_node_a'][Adi_node_in_masked_atoms] + node_count)
                masked_Adi_node_b.append(data['Adi_node_b'][Adi_node_in_masked_atoms] + node_count)
                masked_Adi_node_c.append(data['Adi_node_c'][Adi_node_in_masked_atoms] + node_count)
                masked_Adi_node_d.append(data['Adi_node_d'][Adi_node_in_masked_atoms] + node_count)

            if encoder:
                Bl_node_in_masked_atoms, Ba_node_in_masked_atoms, Adi_node_in_masked_atoms = \
                    self.generate_mask(data, mask_node_i)

                masked_Bl_bond_length.append(data['Bl_bond_length'][Bl_node_in_masked_atoms])
                masked_Ba_bond_angle.append(data['Ba_bond_angle'][Ba_node_in_masked_atoms])
                masked_Adi_angle_dihedral.append(data['Adi_angle_dihedral'][Adi_node_in_masked_atoms])

            if encoder and self.isomorphism:
                isomorphism += [*data['isomorphism']]
                isomorphism_num.append(len(data['isomorphism']))

        if prior:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')

            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')

            feed_dict['Adi_node_a'] = np.concatenate(Adi_node_a, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_b'] = np.concatenate(Adi_node_b, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_c'] = np.concatenate(Adi_node_c, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_d'] = np.concatenate(Adi_node_d, 0).reshape(-1).astype('int64')

            feed_dict['masked_Bl_node_i'] = np.concatenate(masked_Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['masked_Bl_node_j'] = np.concatenate(masked_Bl_node_j, 0).reshape(-1).astype('int64')

            feed_dict['masked_Ba_node_i'] = np.concatenate(masked_Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['masked_Ba_node_j'] = np.concatenate(masked_Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['masked_Ba_node_k'] = np.concatenate(masked_Ba_node_k, 0).reshape(-1).astype('int64')

            feed_dict['masked_Adi_node_a'] = np.concatenate(masked_Adi_node_a, 0).reshape(-1).astype('int64')
            feed_dict['masked_Adi_node_b'] = np.concatenate(masked_Adi_node_b, 0).reshape(-1).astype('int64')
            feed_dict['masked_Adi_node_c'] = np.concatenate(masked_Adi_node_c, 0).reshape(-1).astype('int64')
            feed_dict['masked_Adi_node_d'] = np.concatenate(masked_Adi_node_d, 0).reshape(-1).astype('int64')

        if encoder:
            feed_dict['masked_Bl_bond_length'] = np.concatenate(masked_Bl_bond_length, 0).reshape(-1, 1).astype('float32')
            feed_dict['masked_Ba_bond_angle'] = np.concatenate(masked_Ba_bond_angle, 0).reshape(-1, 1).astype('float32')
            feed_dict['masked_Adi_angle_dihedral'] = np.concatenate(masked_Adi_angle_dihedral, 0).reshape(-1, 1).astype('float32')

        # atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        # self._flat_shapes(atom_bond_graph.node_feat)
        # self._flat_shapes(atom_bond_graph.edge_feat)
        # graph_dict["atom_bond_graph"] = atom_bond_graph
        #
        # bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        # self._flat_shapes(bond_angle_graph.node_feat)
        # self._flat_shapes(bond_angle_graph.edge_feat)
        # graph_dict["bond_angle_graph"] = bond_angle_graph
        #
        # angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)
        # self._flat_shapes(angle_dihedral_graph.node_feat)
        # self._flat_shapes(angle_dihedral_graph.edge_feat)
        # graph_dict['angle_dihedral_graph'] = angle_dihedral_graph

        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        graph_dict["atom_bond_graph"] = masked_atom_bond_graph

        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat)
        graph_dict["bond_angle_graph"] = masked_bond_angle_graph

        masked_angle_dihedral_graph = pgl.Graph.batch(masked_angle_dihedral_graph_list)
        self._flat_shapes(masked_angle_dihedral_graph.node_feat)
        self._flat_shapes(masked_angle_dihedral_graph.edge_feat)
        graph_dict["angle_dihedral_graph"] = masked_angle_dihedral_graph

        batch_list = np.array(batch_list)
        num_nodes = np.array(num_nodes)
        pos_list = np.vstack(pos_list)

        if encoder and self.isomorphism:
            batch = {
                "batch": batch_list,
                "num_nodes": num_nodes,
                "positions": pos_list,
                "isomorphism": np.hstack(isomorphism),
                "isomorphism_num": np.array(isomorphism_num)
            }
            return graph_dict, feed_dict, batch, np.array(masked_nodes)

        batch = {
            "batch": batch_list,
            "num_nodes": num_nodes,
            "positions": pos_list
        }
        return graph_dict, feed_dict, batch, np.array(masked_nodes)

    def __call__(self, batch_data_list):
        prior_data_list = [x[0] for x in batch_data_list]
        decoder_data_list = [x[-2] for x in batch_data_list]
        mol_list = sum([x[-1] for x in batch_data_list], [])

        prior_data_list = sum(prior_data_list, [])
        decoder_data_list = sum(decoder_data_list, [])

        prior_graph, prior_feed, prior_batch, masked_nodes = self.process_data(
            prior_data_list, prior=True, encoder=False
        )

        decoder_graph, decoder_feed, decoder_batch, _ = self.process_data(
            decoder_data_list, prior=False, encoder=False, masked_nodes=masked_nodes
        )

        if self.evaluate:
            return prior_graph, decoder_graph, \
                prior_feed, decoder_feed, \
                prior_batch, decoder_batch, mol_list

        encoder_data_list = [x[1] for x in batch_data_list]
        encoder_data_list = sum(encoder_data_list, [])

        encoder_graph, encoder_feed, encoder_batch, _ = self.process_data(
            encoder_data_list, prior=False, encoder=True, masked_nodes=masked_nodes
        )

        return prior_graph, encoder_graph, decoder_graph, \
            prior_feed, encoder_feed, decoder_feed, \
            prior_batch, encoder_batch, decoder_batch, mol_list


def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(g,
                                 superedge_g,
                                 supersuperedge_g,
                                 target_atom_indices=None,
                                 mask_ratio=None,
                                 mask_value=0,
                                 subgraph_num=None,
                                 version='dgem'):
    """tbd"""

    def get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices):
        """tbd"""
        atomic_num = g.node_feat['atomic_num'].flatten()
        bond_type = g.edge_feat['bond_type'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        return subgraph_str

    g = deepcopy(g)
    N = g.num_nodes
    E = g.num_edges
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)

    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))  # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)
    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    for atom_index in target_atom_indices:
        left_nei_bond_indices = full_bond_indices[g.edges[:, 0] == atom_index]
        right_nei_bond_indices = full_bond_indices[g.edges[:, 1] == atom_index]
        nei_bond_indices = np.append(left_nei_bond_indices, right_nei_bond_indices)
        left_nei_atom_indices = g.edges[left_nei_bond_indices, 1]
        right_nei_atom_indices = g.edges[right_nei_bond_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)

        if version == 'dgem':
            target_label = None
        elif version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(None)

        target_labels.append(target_label)
        Cm_node_i.append([atom_index])
        Cm_node_i.append(nei_atom_indices)
        masked_bond_indices.append(nei_bond_indices)

    target_atom_indices = np.array(target_atom_indices)
    target_labels = np.array(target_labels)
    Cm_node_i = np.concatenate(Cm_node_i, 0)
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)
    for name in g.node_feat:
        g.node_feat[name][Cm_node_i] = mask_value
    for name in g.edge_feat:
        g.edge_feat[name][masked_bond_indices] = mask_value

    # mask superedge_g
    full_superedge_indices = np.arange(superedge_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[superedge_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[superedge_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))

    if len(masked_superedge_indices) != 0:
        masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
        for name in superedge_g.edge_feat:
            superedge_g.edge_feat[name][masked_superedge_indices] = mask_value

    # mask supersuperedge_g
    full_supersuperedge_indices = np.arange(supersuperedge_g.num_edges)
    masked_supersuperedge_indices = []
    for superedge_index in masked_superedge_indices:
        left_indices = full_supersuperedge_indices[supersuperedge_g.edges[:, 0] == superedge_index]
        right_indices = full_supersuperedge_indices[supersuperedge_g.edges[:, 1] == superedge_index]
        masked_supersuperedge_indices.append(np.append(left_indices, right_indices))

    if len(masked_supersuperedge_indices) != 0:
        masked_supersuperedge_indices = np.concatenate(masked_supersuperedge_indices, 0)
        for name in supersuperedge_g.edge_feat:
            supersuperedge_g.edge_feat[name][masked_supersuperedge_indices] = mask_value

    return [g, superedge_g, supersuperedge_g, target_atom_indices, target_labels]
