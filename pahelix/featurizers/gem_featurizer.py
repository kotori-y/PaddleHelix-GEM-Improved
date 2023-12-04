#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
| Featurizers for pretrain-gnn.

| Adapted from https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/utils.py
"""

import numpy as np
import networkx as nx
from copy import deepcopy
import pgl
import rdkit
from rdkit.Chem import AllChem as Chem

from sklearn.metrics import pairwise_distances
import hashlib
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d, mol_to_geognn_graph_data_raw3d
from pahelix.utils.compound_tools import Compound3DKit


def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(
        g,
        superedge_g,
        supersuperedge_g,
        target_atom_indices=None,
        mask_ratio=None,
        mask_value=0,
        subgraph_num=None,
        version='gem'):
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
        masked_size = max(1, int(N * mask_ratio))   # at least 1 atom will be selected.
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

        if version == 'gem':
            subgraph_str = get_subgraph_str(g, atom_index, nei_atom_indices, nei_bond_indices)
            subgraph_id = md5_hash(subgraph_str) % subgraph_num
            target_label = subgraph_id
        else:
            raise ValueError(version)

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


def get_pretrain_bond_angle(superedges, edges, angles):
    """tbd"""
    atoms = []
    bond_angles = []
    for index, superedge in enumerate(superedges):
        edge_a = edges[superedge[0]]
        edge_b = edges[superedge[1]]
        temp = np.hstack([edge_a, edge_b[1:]])

        if temp[0] != temp[-1]:
            atoms.append(temp)
            bond_angles.append(angles[index])

    _atoms = np.array(atoms)

    if len(_atoms) == 0:
        node_i_indices = np.zeros([0, ], 'int64')
        node_j_indices = np.zeros([0, ], 'int64')
        node_k_indices = np.zeros([0, ], 'int64')
        bond_angles = np.zeros([0, ], 'float32')
        return [node_i_indices, node_j_indices, node_k_indices, bond_angles]

    node_i_indices, node_j_indices, node_k_indices = _atoms.T

    return node_i_indices, node_j_indices, node_k_indices, bond_angles


def get_pretrain_angle_dihedral(supersuperedges, superedges, edges, angles):
    """tbd"""
    atoms = []
    dihedral_angles = []
    for index, supersuperedge in enumerate(supersuperedges):
        superedge_a = superedges[supersuperedge[0]]
        superedge_b = superedges[supersuperedge[1]]
        temp = np.hstack([superedge_a, superedge_b[1:]])

        if temp[0] != temp[-1]:
            edge_a = edges[superedge_a[0]]
            edge_d = edges[superedge_b[1]]
            if edge_a.std() == 0 or edge_d.std() == 0:
                continue
            atoms.append(np.hstack([edge_a, edge_d]))
            dihedral_angles.append(angles[index])

    _atoms = np.array(atoms)

    if len(_atoms) == 0:
        node_a_indices = np.zeros([0, ], 'int64')
        node_b_indices = np.zeros([0, ], 'int64')
        node_c_indices = np.zeros([0, ], 'int64')
        node_d_indices = np.zeros([0, ], 'int64')
        dihedral_angles = np.zeros([0, ], 'float32')
        return [node_a_indices, node_b_indices, node_c_indices, node_d_indices, dihedral_angles]

    node_a_indices, node_b_indices, node_c_indices, node_d_indices = _atoms.T
    return [node_a_indices, node_b_indices, node_c_indices, node_d_indices, dihedral_angles]


class GeoPredTransformFn(object):
    """Gen features for downstream model"""
    def __init__(self, pretrain_tasks, mask_ratio, with_provided_3d):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.with_provided_3d = with_provided_3d

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """
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

        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])

        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])

        data['atom_cm5'] = np.array(data.get('cm5', []))
        data['atom_espc'] = np.array(data.get('espc', []))
        data['atom_hirshfeld'] = np.array(data.get('hirshfeld', []))
        data['atom_npa'] = np.array(data.get('npa', []))
        data['bo_bond_order'] = np.array(data.get('bond_order', []))

        return data

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        if self.with_provided_3d:
            mol = raw_data
            if mol is None:
                return None
            smiles = Chem.MolToSmiles(mol)
            print('smiles', smiles)
            data = mol_to_geognn_graph_data_raw3d(mol)
            data['smiles'] = smiles
            data = self.prepare_pretrain_task(data)
            return data

        smiles = raw_data
        print('smiles', smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        data['smiles'] = smiles
        data = self.prepare_pretrain_task(data)
        return data


class GeoPredCollateFn(object):
    """tbd"""
    def __init__(self,
                 atom_names,
                 bond_names,
                 bond_float_names,
                 bond_angle_float_names,
                 dihedral_angle_float_names,
                 pretrain_tasks,
                 mask_ratio,
                 Cm_vocab):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.Cm_vocab = Cm_vocab
        self.bond_angle_float_names = bond_angle_float_names

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):
        """tbd"""
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        angle_dihedral_graph_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_angle_dihedral_graph_list = []
        Cm_node_i = []
        Cm_context_id = []
        Fg_morgan = []
        Fg_daylight = []
        Fg_maccs = []
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
        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []

        bo_bond_order = []
        atom_cm5 = []
        atom_espc = []
        atom_hirshfeld = []
        atom_npa = []

        node_count = 0
        for data in batch_data_list:
            N = len(data[self.atom_names[0]])
            E = len(data['edges'])
            A = len(data['BondAngleGraph_edges'])
            ab_g = pgl.graph.Graph(num_nodes=N,
                                   edges=data['edges'],
                                   node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                                   edge_feat={name: data[name].reshape([-1, 1]) for name in
                                              self.bond_names + self.bond_float_names})
            ba_g = pgl.graph.Graph(num_nodes=E,
                                   edges=data['BondAngleGraph_edges'],
                                   node_feat={},
                                   edge_feat={name: data[name].reshape([-1, 1]) for name in
                                              self.bond_angle_float_names})
            adi_g = pgl.graph.Graph(num_nodes=A,
                                    edges=data['AngleDihedralGraph_edges'],
                                    node_feat={},
                                    edge_feat={name: data[name].reshape([-1, 1]) for name in
                                               self.dihedral_angle_float_names})
            masked_ab_g, masked_ba_g, masked_adi_g, mask_node_i, context_id = mask_context_of_geognn_graph(
                ab_g, ba_g, adi_g, mask_ratio=self.mask_ratio, subgraph_num=self.Cm_vocab)
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            angle_dihedral_graph_list.append(adi_g)
            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_angle_dihedral_graph_list.append(masked_adi_g)
            if 'Cm' in self.pretrain_tasks:
                Cm_node_i.append(mask_node_i + node_count)
                Cm_context_id.append(context_id)
            if 'Fg' in self.pretrain_tasks:
                Fg_morgan.append(data['morgan_fp'])
                Fg_daylight.append(data['daylight_fg_counts'])
                Fg_maccs.append(data['maccs_fp'])
            if 'Bar' in self.pretrain_tasks:
                Ba_node_i.append(data['Ba_node_i'] + node_count)
                Ba_node_j.append(data['Ba_node_j'] + node_count)
                Ba_node_k.append(data['Ba_node_k'] + node_count)
                Ba_bond_angle.append(data['Ba_bond_angle'])
            if 'Dir' in self.pretrain_tasks:
                Adi_node_a.append(data['Adi_node_a'] + node_count)
                Adi_node_b.append(data['Adi_node_b'] + node_count)
                Adi_node_c.append(data['Adi_node_c'] + node_count)
                Adi_node_d.append(data['Adi_node_d'] + node_count)
                Adi_angle_dihedral.append(data['Adi_angle_dihedral']),
            if 'Blr' in self.pretrain_tasks:
                Bl_node_i.append(data['Bl_node_i'] + node_count)
                Bl_node_j.append(data['Bl_node_j'] + node_count)
                Bl_bond_length.append(data['Bl_bond_length'])
                if 'wiberg' in self.pretrain_tasks:
                    bo_bond_order.append(data['bo_bond_order'])
            if 'Adc' in self.pretrain_tasks:
                Ad_node_i.append(data['Ad_node_i'] + node_count)
                Ad_node_j.append(data['Ad_node_j'] + node_count)
                Ad_atom_dist.append(data['Ad_atom_dist'])
            if 'cm5' in self.pretrain_tasks:
                atom_cm5.append(data['atom_cm5'])
            if 'espc' in self.pretrain_tasks:
                atom_espc.append(data['atom_espc'])
            if 'hirshfeld' in self.pretrain_tasks:
                atom_hirshfeld.append(data['atom_hirshfeld'])
            if 'npa' in self.pretrain_tasks:
                atom_npa.append(data['atom_npa'])

            node_count += N

        graph_dict = {}
        feed_dict = {}

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        graph_dict['atom_bond_graph'] = atom_bond_graph

        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        graph_dict['bond_angle_graph'] = bond_angle_graph

        angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)
        self._flat_shapes(angle_dihedral_graph.node_feat)
        self._flat_shapes(angle_dihedral_graph.edge_feat)
        graph_dict['angle_dihedral_graph'] = angle_dihedral_graph

        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph

        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat)
        graph_dict['masked_bond_angle_graph'] = masked_bond_angle_graph

        masked_angle_dihedral_graph = pgl.Graph.batch(masked_angle_dihedral_graph_list)
        self._flat_shapes(masked_angle_dihedral_graph.node_feat)
        self._flat_shapes(masked_angle_dihedral_graph.edge_feat)
        graph_dict['masked_angle_dihedral_graph'] = masked_angle_dihedral_graph

        if 'Cm' in self.pretrain_tasks:
            feed_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')
        if 'Fg' in self.pretrain_tasks:
            feed_dict['Fg_morgan'] = np.array(Fg_morgan, 'float32')
            feed_dict['Fg_daylight'] = (np.array(Fg_daylight) > 0).astype('float32')  # >1: 1x
            feed_dict['Fg_maccs'] = np.array(Fg_maccs, 'float32')
        if 'Bar' in self.pretrain_tasks:
            feed_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
            feed_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')
        if 'Dir' in self.pretrain_tasks:
            feed_dict['Adi_node_a'] = np.concatenate(Adi_node_a, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_b'] = np.concatenate(Adi_node_b, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_c'] = np.concatenate(Adi_node_c, 0).reshape(-1).astype('int64')
            feed_dict['Adi_node_d'] = np.concatenate(Adi_node_c, 0).reshape(-1).astype('int64')
            feed_dict['Adi_angle_dihedral'] = np.concatenate(Adi_angle_dihedral, 0).reshape(-1, 1).astype('float32')
        if 'Blr' in self.pretrain_tasks:
            feed_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')
            if 'wiberg' in self.pretrain_tasks:
                feed_dict['bo_bond_order'] = np.concatenate(bo_bond_order, 0).reshape(-1, 1).astype('float32')
        if 'Adc' in self.pretrain_tasks:
            feed_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
            feed_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
            feed_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')
        if 'cm5' in self.pretrain_tasks:
            feed_dict['atom_cm5'] = np.hstack(atom_cm5).astype('float32')
        if 'espc' in self.pretrain_tasks:
            feed_dict['atom_espc'] = np.hstack(atom_espc).astype('float32')
        if 'hirshfeld' in self.pretrain_tasks:
            feed_dict['atom_hirshfeld'] = np.hstack(atom_hirshfeld).astype('float32')
        if 'npa' in self.pretrain_tasks:
            feed_dict['atom_npa'] = np.hstack(atom_npa).astype('float32')

        return graph_dict, feed_dict

