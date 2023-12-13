import pickle

import numpy as np
import paddle
from rdkit import Chem
import pgl
from sklearn.metrics import pairwise_distances

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean, set_rdmol_positions
from pahelix.datasets.inmemory_dataset import InMemoryDataset
from pahelix.featurizers.gem_featurizer import get_pretrain_bond_angle, get_pretrain_angle_dihedral
from pahelix.utils import load_json_config
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_raw3d, mol_to_geognn_graph_data_MMFF3d, \
    mol_to_geognn_graph_data


def move2origin(poses, batch, num_nodes):
    dim_size = batch.max() + 1
    index = paddle.to_tensor(batch)
    poses_mean = scatter_mean(poses, index, 0, dim_size)
    _poses_mean = poses_mean.numpy().repeat(num_nodes, axis=0)
    _poses_mean = paddle.to_tensor(_poses_mean, dtype=poses_mean.dtype)
    return poses - _poses_mean


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

    def __init__(self, n_noise_mol=10):
        self.n_noise_mol = n_noise_mol

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

    def __call__(self, mol):
        """
        """
        gt_pos = mol.GetConformer().GetPositions()

        prior_pos = np.random.uniform(-1, 1, size=gt_pos.shape)
        encoder_pos_list = [gt_pos + np.random.uniform(0, 0.5, size=gt_pos.shape) for _ in range(self.n_noise_mol)]
        decoder_pos = gt_pos

        smiles = Chem.MolToSmiles(mol)
        print('smiles', smiles)

        prior_data = mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, prior_pos), only_atom_bond=True)
        encoder_data_list = [
            mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, pos))
            for pos in encoder_pos_list
        ]

        decoder_gt_data = mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, decoder_pos))
        decoder_gt_data = self.prepare_pretrain_task(decoder_gt_data)

        decoder_data = mol_to_geognn_graph_data_raw3d(set_rdmol_positions(mol, decoder_pos), only_atom_bond=True)

        prior_data['smiles'] = smiles
        decoder_data['smiles'] = smiles
        decoder_data['smiles'] = smiles

        # return prior_data, encoder_data_list, decoder_data
        return prior_data, encoder_data_list, decoder_data, decoder_gt_data

    # def __call__(self, mol):
    #     prior_data = []
    #     encoder_data = []
    #     decoder_data = []
    #
    #     gt_pos = mol.GetConformer().GetPositions()
    #     # if self.add_noise:
    #     #     noise = np.random.uniform(-1, 1, size=gt_pos.shape)
    #     #     gt_pos += noise
    #
    #     # prior_pos = np.random.uniform(-0.5, 0.5, gt_pos.shape)
    #     # prior_pos = gt_pos
    #
    #     if not self.is_inference:
    #         data = mol_to_geognn_graph_data(mol, gt_pos - gt_pos.mean(axis=0), dir_type='HT', only_atom_bond=False)
    #         # data = mol_to_geognn_graph_data(mol, gt_pos, dir_type='HT', only_atom_bond=False)
    #         node_i, node_j, node_k, bond_angles = \
    #             get_pretrain_bond_angle(data['BondAngleGraph_edges'], data['edges'], data['bond_angle'])
    #
    #         data['Ba_node_i'] = node_i
    #         data['Ba_node_j'] = node_j
    #         data['Ba_node_k'] = node_k
    #         data['Ba_bond_angle'] = bond_angles
    #
    #         data['Bl_node_i'] = np.array(data['edges'][:, 0])
    #         data['Bl_node_j'] = np.array(data['edges'][:, 1])
    #         data['Bl_bond_length'] = np.array(data['bond_length'])
    #     else:
    #         data = None
    #
    #     # prior_data = mol_to_geognn_graph_data(mol, None, dir_type='HT', only_atom_bond=True, with_distance=False)
    #
    #     return [data, gt_pos, mol]


class ConfGenTaskCollateFn(object):
    def __init__(
            self,
            atom_names,
            bond_names,
            bond_float_names,
            bond_angle_float_names,
            dihedral_angle_float_names,
            pretrain_tasks
         ):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names

        self.pretrain_tasks = pretrain_tasks

    def _flat_shapes(self, d):
        for name in d:
            d[name] = d[name].reshape([-1])

    def process_data(self, data_list, only_atom_bond=False, pretrain=False):
        if pretrain: assert not only_atom_bond

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

        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []

        bo_bond_order = []
        atom_cm5 = []
        atom_espc = []
        atom_hirshfeld = []
        atom_npa = []

        node_count = 0
        for data in data_list:
            N = len(data[self.atom_names[0]])

            if pretrain:
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
                    Adi_angle_dihedral.append(data['Adi_angle_dihedral'])

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

                continue

            if only_atom_bond:
                ab_g = pgl.Graph(
                    num_nodes=N,
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names}
                )

                atom_bond_graph_list.append(ab_g)
            else:
                E = len(data['edges'])
                A = len(data['BondAngleGraph_edges'])

                ab_g = pgl.Graph(
                    num_nodes=N,
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names}
                )

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

                atom_bond_graph_list.append(ab_g)
                bond_angle_graph_list.append(ba_g)
                angle_dihedral_graph_list.append(adi_g)

        if pretrain:
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
                feed_dict['Adi_angle_dihedral'] = np.concatenate(Adi_angle_dihedral, 0).reshape(-1, 1).astype(
                    'float32')
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

            return feed_dict

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

        return graph_dict

    def __call__(self, batch_data_list):
        prior_data = [x[0] for x in batch_data_list]
        encoder_data_list = [x[1] for x in batch_data_list]
        decoder_data = [x[2] for x in batch_data_list]
        decoder_gt_data = [x[3] for x in batch_data_list]

        prior_data_list = np.repeat(prior_data, len(encoder_data_list[0]))
        decoder_data_list = np.repeat(decoder_data, len(encoder_data_list[0]))
        decoder_gt_data_list = np.repeat(decoder_gt_data, len(encoder_data_list[0]))
        encoder_data_list = sum(encoder_data_list, [])

        prior_graph = self.process_data(prior_data_list, only_atom_bond=True)
        encoder_graph = self.process_data(encoder_data_list, only_atom_bond=False)
        decoder_graph = self.process_data(decoder_data_list, only_atom_bond=True)
        decoder_feed = self.process_data(decoder_gt_data_list, only_atom_bond=False, pretrain=True)

        return prior_graph, encoder_graph, decoder_graph, decoder_feed
