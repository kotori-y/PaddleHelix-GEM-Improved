import copy

import numpy as np
import paddle
import paddle.nn as nn
import pgl

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.loss.e3_loss import compute_loss
from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel, GeoGNNModelOld
from pahelix.networks.basic_block import MLP, MLPwoLastAct
from pahelix.utils.compound_tools import mol_to_geognn_graph_data, mol_to_geognn_graph_data_MMFF3d, Compound3DKit

from rdkit import Chem
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class ConfGenModel(nn.Layer):
    def __init__(self, prior_config, encoder_config, decoder_config, down_config, aux_config=None, recycle=0, init_model=None):
        super(ConfGenModel, self).__init__()
        assert recycle >= 0

        self.prior_config = prior_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.down_config = down_config

        # prior p(z|G',H')
        self.prior_gnn = GNNModel(prior_config)
        if init_model:
            self.prior_gnn.set_state_dict(paddle.load(init_model))
            print(f'load model from {init_model}')
        self.prior_pos = MLPwoLastAct(
            layer_num=3,
            in_size=self.prior_gnn.graph_dim,
            hidden_size=down_config['prior_hidden_size'],
            out_size=3,
            act=down_config['act'],
            dropout_rate=0.0
        )

        # encoder q(z|G,H)
        self.encoder_gnn = GNNModel(encoder_config)
        if init_model:
            self.encoder_gnn.set_state_dict(paddle.load(init_model))
            print(f'load model from {init_model}')
        self.encoder_head = MLP(
            layer_num=down_config['layer_num'],
            in_size=self.encoder_gnn.graph_dim,
            hidden_size=down_config['encoder_hidden_size'],
            out_size=decoder_config['embed_dim'] * 2,
            act=down_config['act'],
            dropout_rate=down_config['dropout_rate']
        )
        # encoder

        # decoder q(X|G,z)
        self.decoder_gnn = nn.LayerList()
        self.decoder_pos = nn.LayerList()

        for _ in range(recycle + 1):
            decoder = GNNModel(decoder_config)
            if init_model:
                decoder.set_state_dict(paddle.load(init_model))
            decoder_pos = MLPwoLastAct(
                layer_num=3,
                in_size=decoder.graph_dim,
                hidden_size=down_config['decoder_hidden_size'],
                out_size=3,
                act=down_config['act'],
                dropout_rate=0.0
            )
            self.decoder_gnn.append(decoder)
            self.decoder_pos.append(decoder_pos)

        self.decoder_blr = MLP(
            layer_num=2,
            hidden_size=down_config['decoder_hidden_size'],
            act=down_config['act'],
            in_size=decoder_config['embed_dim'] * 2,
            out_size=1,
            dropout_rate=0.2
        )
        self.decoder_bar = MLP(
            layer_num=2,
            hidden_size=down_config['decoder_hidden_size'],
            act=down_config['act'],
            in_size=decoder_config['embed_dim'] * 3,
            out_size=1,
            dropout_rate=0.2
        )
        # decoder q(X|G,z)

        # self.Blr_loss = nn.SmoothL1Loss()
        # self.decoder_norm = nn.LayerNorm(decoder.graph_dim)
        # decoder

    def forward(self, atom_bond_graphs, bond_angle_graphs, batch, gt_pos_list, args, feed_dict=None, sample=False):
        assert (self.training and feed_dict is not None) or not self.training

        extra_output = {}
        cur_pos = []
        for mol in copy.deepcopy(batch["mols"]):
            if len(mol.GetAtoms()) <= 400:
                _, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=1)
            else:
                atom_poses = Compound3DKit.get_2d_atom_poses(mol)

            atom_poses = np.array(atom_poses)
            cur_pos.append(atom_poses)

        cur_pos = paddle.to_tensor(np.vstack(cur_pos), dtype=paddle.float32)

        # n_atoms = batch["num_nodes"].sum()
        # init_pos = paddle.randn(shape=(n_atoms, 3), dtype=paddle.float32)

        # prior p(z|G)
        prior_atom_bond_graphs = ConfGenModel.update_graph(self.prior_config, batch, cur_pos, only_atom_bond=True)
        prior_node_repr, _, prior_graph_repr = self.prior_gnn(prior_atom_bond_graphs)

        # cur_pos = self.prior_pos(prior_node_repr)
        delta_pos = self.prior_pos(prior_node_repr)
        cur_pos = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"], batch["num_nodes"])
        extra_output["prior_pos_list"] = [cur_pos]
        # prior

        # encoder q(z|G, H)
        if not sample:
            _, _, graph_repr = self.encoder_gnn(atom_bond_graphs)
            latent = self.encoder_head(graph_repr)
            latent_mean, latent_logstd = paddle.chunk(latent, chunks=2, axis=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd
            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = paddle.randn(prior_graph_repr.shape)
        z = paddle.to_tensor(np.repeat(z.numpy(), batch["num_nodes"], axis=0))

        # paddle.randn(prior_graph_repr.shape)

        # decoder q(X|G,z)
        pos_list = []

        for i, layer in enumerate(self.decoder_gnn):
            atom_bond_graphs = ConfGenModel.update_graph(self.decoder_config, batch, cur_pos, only_atom_bond=True)
            node_repr, _, _ = layer(atom_bond_graphs, z=z)
            # cur_pos = self.decoder_pos[i](node_repr)
            delta_pos = self.decoder_pos[i](node_repr)
            cur_pos = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"], batch["num_nodes"])
            pos_list.append(cur_pos)

        # if self.training:
        #     node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        #     node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        #     node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        #     node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        #     extra_output["bar_list"] = self.decoder_bar(node_ijk_repr)
        #
        #     node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        #     node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        #     node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        #     extra_output["blr_list"] = self.decoder_blr(node_ij_repr)
        if self.training:
            loss, loss_dict = compute_loss(extra_output, feed_dict, gt_pos_list, pos_list, batch, args)
            return loss, loss_dict, pos_list

        return pos_list


    # @staticmethod
    # def update_iso(pos_y, pos_x, batch):
    #     with paddle.no_grad():
    #         pre_nodes = 0
    #         num_nodes = batch.n_nodes
    #         isomorphisms = batch.isomorphisms
    #         new_idx_x = []
    #         for i in range(batch.num_graphs):
    #             cur_num_nodes = num_nodes[i]
    #             current_isomorphisms = [
    #                 torch.LongTensor(iso).to(pos_x.device) for iso in isomorphisms[i]
    #             ]
    #             if len(current_isomorphisms) == 1:
    #                 new_idx_x.append(current_isomorphisms[0] + pre_nodes)
    #             else:
    #                 pos_y_i = pos_y[pre_nodes: pre_nodes + cur_num_nodes]
    #                 pos_x_i = pos_x[pre_nodes: pre_nodes + cur_num_nodes]
    #                 pos_y_mean = torch.mean(pos_y_i, dim=0, keepdim=True)
    #                 pos_x_mean = torch.mean(pos_x_i, dim=0, keepdim=True)
    #                 pos_x_list = []
    #
    #                 for iso in current_isomorphisms:
    #                     pos_x_list.append(torch.index_select(pos_x_i, 0, iso))
    #                 total_iso = len(pos_x_list)
    #                 pos_y_i = pos_y_i.repeat(total_iso, 1)
    #                 pos_x_i = torch.cat(pos_x_list, dim=0)
    #                 min_idx = GNN.alignment_loss_iso_onegraph(
    #                     pos_y_i,
    #                     pos_x_i,
    #                     pos_y_mean,
    #                     pos_x_mean,
    #                     num_nodes=cur_num_nodes,
    #                     total_iso=total_iso,
    #                 )
    #                 new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
    #             pre_nodes += cur_num_nodes
    #
    #         return torch.cat(new_idx_x, dim=0)

    @staticmethod
    def update_graph(model_config, batch, cur_pos, with_distance=True, only_atom_bond=False):
        data_list = []
        prev_node = 0
        for i, mol in enumerate(batch["mols"]):
            n_atom = batch["num_nodes"][i]
            tmp_pos = cur_pos[prev_node: n_atom + prev_node]
            data_list.append(mol_to_geognn_graph_data(
                mol, atom_poses=tmp_pos, dir_type="HT",
                only_atom_bond=only_atom_bond, with_distance=with_distance
            ))
            prev_node += n_atom

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        # angle_dihedral_graph_list = []

        for i, data in enumerate(data_list):
            ab_g = pgl.Graph(
                num_nodes=len(data[model_config["atom_names"][0]]),
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in model_config["atom_names"]},
                edge_feat={name: data[name].reshape([-1, 1]) for name in
                           model_config["bond_names"] + model_config["bond_float_names"]})
            atom_bond_graph_list.append(ab_g)

            if not only_atom_bond:
                ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in
                               model_config["bond_angle_float_names"]})
                # adi_g = pgl.graph.Graph(
                #     num_nodes=len(data['BondAngleGraph_edges']),
                #     edges=data['AngleDihedralGraph_edges'],
                #     node_feat={},
                #     edge_feat={name: data[name].reshape([-1, 1]) for name in
                #                model_config["dihedral_angle_float_names"]})
                bond_angle_graph_list.append(ba_g)
                # angle_dihedral_graph_list.append(adi_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        if not only_atom_bond:
            bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
            # angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)

        # TODO: reshape due to pgl limitations on the shape
        ConfGenModel._flat_shapes(atom_bond_graph.node_feat)
        ConfGenModel._flat_shapes(atom_bond_graph.edge_feat)
        if not only_atom_bond:
            ConfGenModel._flat_shapes(bond_angle_graph.node_feat)
            ConfGenModel._flat_shapes(bond_angle_graph.edge_feat)
            # ConfGenModel._flat_shapes(angle_dihedral_graph.node_feat)
            # ConfGenModel._flat_shapes(angle_dihedral_graph.edge_feat)
            return atom_bond_graph.tensor(), bond_angle_graph.tensor()  # angle_dihedral_graph.tensor()

        return atom_bond_graph.tensor()

    @staticmethod
    def _flat_shapes(d):
        for name in d:
            d[name] = d[name].reshape([-1])

    @staticmethod
    def move2origin(poses, batch, num_nodes):
        dim_size = batch.max() + 1
        index = paddle.to_tensor(batch)
        poses_mean = scatter_mean(poses, index, 0, dim_size)
        _poses_mean = poses_mean.numpy().repeat(num_nodes, axis=0)
        poses_mean = paddle.to_tensor(_poses_mean, dtype=poses_mean.dtype)

        return poses - poses_mean

    @staticmethod
    def reparameterization(mean, log_std):
        std = paddle.exp(0.5 * log_std)
        # eps = paddle.randn(std.size()).to(mean)
        eps = paddle.randn(std.shape)
        # z = mean + std * epsilon
        return mean + std * eps
