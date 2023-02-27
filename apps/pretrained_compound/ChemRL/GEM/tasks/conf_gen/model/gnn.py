import copy

import numpy as np
import paddle
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel
from pahelix.networks.basic_block import MLP, MLPwoLastAct
from pahelix.utils.compound_tools import mol_to_geognn_graph_data

from rdkit import Chem


class ConfGenModel(nn.Layer):
    def __init__(self, model_config, compound_encoder_config, recycle=0):

        super(ConfGenModel, self).__init__()

        self.compound_encoder_config = compound_encoder_config
        assert recycle >= 0

        # prior
        self.prior_gnn = nn.LayerList()
        self.prior_pos = nn.LayerList()
        for _ in range(recycle + 1):
            prior_encoder = GeoGNNModel(compound_encoder_config)
            prior_pos = MLPwoLastAct(
                model_config['layer_num'],
                in_size=prior_encoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=3,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )
            self.prior_pos.append(prior_pos)
            self.prior_gnn.append(prior_encoder)
        # self.prior_norm = nn.LayerNorm(prior_encoder.graph_dim)
        # prior

        # encoder
        self.encoder_gnn = nn.LayerList()
        for _ in range(recycle + 1):
            encoder = GeoGNNModel(compound_encoder_config)
            self.encoder_gnn.append(encoder)
        self.encoder_head = MLPwoLastAct(
            model_config['layer_num'],
            in_size=compound_encoder_config['embed_dim'],
            hidden_size=model_config['hidden_size'],
            out_size=compound_encoder_config['embed_dim'] * 2,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate']
        )
        # encoder

        # decoder
        self.decoder_gnn = nn.LayerList()
        self.decoder_pos = nn.LayerList()
        for _ in range(recycle + 1):
            decoder = GeoGNNModel(compound_encoder_config)
            decoder_pos = MLPwoLastAct(
                model_config['layer_num'],
                in_size=decoder.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=3,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )
            self.decoder_gnn.append(decoder)
            self.decoder_pos.append(decoder_pos)
        # self.decoder_norm = nn.LayerNorm(decoder.graph_dim)
        # decoder

    def forward(self,
                prior_atom_bond_graphs, prior_bond_angle_graph, prior_angle_dihedral_graph,
                atom_bond_graphs, bond_angle_graph, angle_dihedral_graph,
                prior_poses, batch, sample=False):
        # CCLKSILAUUULCQ-RITPCOANSA-N
        extra_output = {}

        # prior conf
        # total_nodes = batch["num_nodes"].sum()
        cur_pos = prior_poses
        pos_list = []

        prior_node_repr, prior_edge_repr, prior_angle_repr = None, None, None
        for i, layer in enumerate(self.prior_gnn):
            prior_node_repr, prior_edge_repr, prior_angle_repr, prior_graph_repr = layer(
                prior_atom_bond_graphs, prior_bond_angle_graph, prior_angle_dihedral_graph,
                atom_residual=prior_node_repr, bond_residual=prior_edge_repr, angle_residual=prior_angle_repr
            )
            delta_pos = self.prior_pos[i](prior_node_repr)
            cur_pos, _ = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"])
            pos_list.append(cur_pos)
            if i != len(self.prior_gnn) - 1:
                prior_atom_bond_graphs, prior_bond_angle_graph, prior_angle_dihedral_graph = \
                    self.update_graph(self.compound_encoder_config, batch, cur_pos)
        else:
            extra_output["prior_pos_list"] = pos_list
            prior_output = [prior_node_repr, prior_edge_repr, prior_angle_repr]
        # prior encoder

        if not sample:
            # encoder
            node_repr, edge_repr, angle_repr = None, None, None
            for i, layer in enumerate(self.encoder_gnn):
                node_repr, edge_repr, angle_repr, graph_repr = layer(
                    atom_bond_graphs, bond_angle_graph, angle_dihedral_graph,
                    atom_residual=node_repr, bond_residual=edge_repr, angle_residual=angle_repr
                )

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

        node_repr, edge_repr, angle_repr = prior_output
        for i, layer in enumerate(self.decoder_gnn):
            atom_bond_graphs, bond_angle_graph, angle_dihedral_graph = \
                ConfGenModel.update_graph(self.compound_encoder_config, batch, cur_pos, only_atom_bond=False)

            node_repr, edge_repr, angle_repr, graph_repr = layer(
                atom_bond_graphs, bond_angle_graph, angle_dihedral_graph,
                atom_residual=node_repr, bond_residual=edge_repr, angle_residual=angle_repr, z=z
            )
            delta_pos = self.decoder_pos[i](node_repr)
            cur_pos, _ = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"])
            pos_list.append(cur_pos)

        return extra_output, pos_list

    @staticmethod
    def update_graph(compound_encoder_config, batch, cur_pos, only_atom_bond=False):
        data_list = []
        prev_node = 0
        for i, mol in enumerate(batch["mols"]):
            n_atom = batch["num_nodes"][i]
            tmp_pos = cur_pos[prev_node: n_atom + prev_node]
            data_list.append(mol_to_geognn_graph_data(mol, atom_poses=tmp_pos, dir_type="HT", only_atom_bond=only_atom_bond))
            prev_node += n_atom

        atom_bond_graph_list = []
        bond_angle_graph_list = []
        angle_dihedral_graph_list = []

        for i, data in enumerate(data_list):
            ab_g = pgl.Graph(
                num_nodes=len(data[compound_encoder_config["atom_names"][0]]),
                edges=data['edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in compound_encoder_config["atom_names"]},
                edge_feat={name: data[name].reshape([-1, 1]) for name in
                           compound_encoder_config["bond_names"] + compound_encoder_config["bond_float_names"]})
            atom_bond_graph_list.append(ab_g)

            if not only_atom_bond:
                ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in
                               compound_encoder_config["bond_angle_float_names"]})
                adi_g = pgl.graph.Graph(
                    num_nodes=len(data['BondAngleGraph_edges']),
                    edges=data['AngleDihedralGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in
                               compound_encoder_config["dihedral_angle_float_names"]})
                bond_angle_graph_list.append(ba_g)
                angle_dihedral_graph_list.append(adi_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        if not only_atom_bond:
            bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
            angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)

        # TODO: reshape due to pgl limitations on the shape
        ConfGenModel._flat_shapes(atom_bond_graph.node_feat)
        ConfGenModel._flat_shapes(atom_bond_graph.edge_feat)
        if not only_atom_bond:
            ConfGenModel._flat_shapes(bond_angle_graph.node_feat)
            ConfGenModel._flat_shapes(bond_angle_graph.edge_feat)
            ConfGenModel._flat_shapes(angle_dihedral_graph.node_feat)
            ConfGenModel._flat_shapes(angle_dihedral_graph.edge_feat)
            return atom_bond_graph.tensor(), bond_angle_graph.tensor(), angle_dihedral_graph.tensor()

        return atom_bond_graph.tensor()

    @staticmethod
    def _flat_shapes(d):
        for name in d:
            d[name] = d[name].reshape([-1])

    @staticmethod
    def move2origin(poses, batch):
        new_pos = []
        pos_mean = []

        for i in range(batch.max() + 1):
            pos = poses[batch == i]
            _mean = pos.mean(axis=0)

            new_pos.append(pos - _mean)
            pos_mean.append(_mean)

        return paddle.concat(new_pos), paddle.stack(pos_mean)

    @staticmethod
    def reparameterization(mean, log_std):
        std = paddle.exp(log_std)
        epsilon = paddle.randn(std.shape)
        z = mean + std * epsilon
        return z
