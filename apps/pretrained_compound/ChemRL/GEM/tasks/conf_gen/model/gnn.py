import copy

import numpy as np
import paddle
import paddle.nn as nn
import pgl

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel, GeoGNNModelOld
from pahelix.networks.basic_block import MLP, MLPwoLastAct
from pahelix.utils.compound_tools import mol_to_geognn_graph_data

from rdkit import Chem


class ConfGenModel(nn.Layer):
    def __init__(self, encoder_config, decoder_config, down_config, aux_config, recycle=0):
        super(ConfGenModel, self).__init__()
        assert recycle >= 0

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.down_config = down_config

        # prior p(z|G)
        self.prior_gnn = nn.LayerList()
        self.prior_pos = nn.LayerList()
        for _ in range(int(recycle / 2) + 1):
            prior_encoder = GNNModel(encoder_config)
            prior_pos = MLPwoLastAct(
                down_config['layer_num'],
                in_size=prior_encoder.graph_dim,
                hidden_size=down_config['encoder_hidden_size'],
                out_size=3,
                act=down_config['act'],
                dropout_rate=down_config['dropout_rate']
            )
            self.prior_pos.append(prior_pos)
            self.prior_gnn.append(prior_encoder)
        # self.prior_norm = nn.LayerNorm(prior_encoder.graph_dim)
        # prior

        # encoder
        self.encoder_gnn = nn.LayerList()
        for _ in range(1):
            encoder = GNNModel(encoder_config)
            self.encoder_gnn.append(encoder)
        self.encoder_head = MLPwoLastAct(
            down_config['layer_num'],
            in_size=encoder_config['embed_dim'],
            hidden_size=down_config['encoder_hidden_size'],
            out_size=decoder_config['embed_dim'] * 2,
            act=down_config['act'],
            dropout_rate=down_config['dropout_rate']
        )
        # encoder

        # decoder
        self.decoder_gnn = nn.LayerList()
        self.decoder_pos = nn.LayerList()
        for _ in range(recycle + 1):
            decoder = GeoGNNModelOld(decoder_config)
            decoder_pos = MLPwoLastAct(
                down_config['layer_num'],
                in_size=decoder.graph_dim,
                hidden_size=down_config['decoder_hidden_size'],
                out_size=3,
                act=down_config['act'],
                dropout_rate=down_config['dropout_rate']
            )
            self.decoder_gnn.append(decoder)
            self.decoder_pos.append(decoder_pos)
        # self.decoder_norm = nn.LayerNorm(decoder.graph_dim)
        # decoder

    def forward(self, atom_bond_graphs, batch, sample=False):
        extra_output = {}

        n_atoms = atom_bond_graphs.num_nodes
        # prior encoder
        prior_poses = paddle.uniform((n_atoms, 3), min=0, max=1)
        cur_pos = prior_poses
        pos_list = []

        for i, layer in enumerate(self.prior_gnn):
            prior_atom_bond_graphs = self.update_graph(self.encoder_config, batch, cur_pos, only_atom_bond=True)
            prior_node_repr, prior_edge_repr, prior_graph_repr = layer(prior_atom_bond_graphs)

            delta_pos = self.prior_pos[i](prior_node_repr)
            cur_pos = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"], batch["num_nodes"])
            pos_list.append(cur_pos)

        else:
            extra_output["prior_pos_list"] = pos_list
        # prior encoder

        if not sample:
            # encoder
            for i, layer in enumerate(self.encoder_gnn):
                node_repr, edge_repr, graph_repr = layer(atom_bond_graphs)

            aggregated_feat = graph_repr
            latent = self.encoder_head(aggregated_feat)
            latent_mean, latent_logstd = paddle.chunk(latent, chunks=2, axis=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd

            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = paddle.randn(prior_graph_repr.shape)
        z = paddle.to_tensor(np.repeat(z.numpy(), batch["num_nodes"], axis=0))

        # decoder
        cur_pos = pos_list[-1]
        pos_list = []

        for i, layer in enumerate(self.decoder_gnn):
            atom_bond_graphs, bond_angle_graph = \
                ConfGenModel.update_graph(self.decoder_config, batch, cur_pos, only_atom_bond=False)

            node_repr, edge_repr, graph_repr = layer(
                atom_bond_graphs, bond_angle_graph, z=z
            )
            delta_pos = self.decoder_pos[i](node_repr)
            cur_pos = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"], batch["num_nodes"])
            pos_list.append(cur_pos)

        return extra_output, pos_list

    @staticmethod
    def update_graph(model_config, batch, cur_pos, only_atom_bond=False):
        data_list = []
        prev_node = 0
        for i, mol in enumerate(batch["mols"]):
            n_atom = batch["num_nodes"][i]
            tmp_pos = cur_pos[prev_node: n_atom + prev_node]
            data_list.append(mol_to_geognn_graph_data(mol, atom_poses=tmp_pos, dir_type="HT", only_atom_bond=only_atom_bond))
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
        std = paddle.exp(log_std)
        epsilon = paddle.randn(std.shape)
        z = mean + std * epsilon
        return z
