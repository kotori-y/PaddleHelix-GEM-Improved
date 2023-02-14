import numpy as np
import paddle
import paddle.nn as nn
import pgl

from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel
from pahelix.networks.basic_block import MLP
from pahelix.utils.compound_tools import mol_to_geognn_graph_data


class ConfGenModel(nn.Layer):
    def __init__(self, model_config, compound_encoder_config,
                 prior_model: GNNModel, encoder_model: GNNModel, decoder_model: GeoGNNModel):

        super(ConfGenModel, self).__init__()

        self.compound_encoder_config = compound_encoder_config

        # prior
        self.prior_gnn = prior_model
        self.prior_pos = nn.LayerList()
        for _ in range(prior_model.layer_num):
            prior_pos = MLP(
                model_config['layer_num'],
                in_size=prior_model.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=3,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )
            self.prior_pos.append(prior_pos)
        # prior

        # encoder
        self.encoder_gnn = encoder_model
        self.encoder_head = MLP(
            model_config['layer_num'],
            in_size=encoder_model.graph_dim,
            hidden_size=model_config['hidden_size'],
            out_size=encoder_model.graph_dim * 2,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate']
        )
        # encoder

        # decoder
        self.decoder_gnn = decoder_model
        self.decoder_pos = nn.LayerList()
        for _ in range(decoder_model.layer_num):
            decoder_pos = MLP(
                model_config['layer_num'],
                in_size=decoder_model.graph_dim,
                hidden_size=model_config['hidden_size'],
                out_size=3,
                act=model_config['act'],
                dropout_rate=model_config['dropout_rate']
            )
            self.decoder_pos.append(decoder_pos)
        # self.decoder_norm = nn.LayerNorm(self.decoder.graph_dim)
        # decoder

    def forward(self, prior_atom_bond_graphs, atom_bond_graphs, prior_poses, batch, sample=False):

        extra_output = {}

        # prior conf
        # total_nodes = batch["num_nodes"].sum()
        cur_pos = prior_poses
        pos_list = []

        node_hidden, edge_hidden, graph_hidden, node_hidden_list = self.prior_gnn(prior_atom_bond_graphs)
        for i, node_repr in enumerate(node_hidden_list):
            delta_pos = self.prior_pos[i](node_repr)
            cur_pos, _ = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"])
            # cur_pos = cur_pos + delta_pos
            pos_list.append(cur_pos)
        else:
            extra_output["prior_pos_list"] = pos_list
            prior_output = [node_hidden, edge_hidden, graph_hidden]
        # prior encoder

        if not sample:
            # encoder
            _, _, graph_repr, _ = self.encoder_gnn(atom_bond_graphs)

            aggregated_feat = graph_repr
            latent = self.encoder_head(aggregated_feat)
            latent_mean, latent_logstd = paddle.chunk(latent, chunks=2, axis=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd

            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = paddle.randn(graph_hidden.shape)
        z = paddle.to_tensor(np.repeat(z.numpy(), batch["num_nodes"], axis=0))

        # decoder
        cur_pos = pos_list[-1]
        pos_list = []

        new_data = []
        n = 0
        for mol in batch["mols"]:
            n_atom = mol.GetConformer().GetPositions().shape[0]
            tmp_pos = cur_pos[n: n_atom + n]
            new_data.append(mol_to_geognn_graph_data(mol, atom_poses=tmp_pos, dir_type="HT"))
            n += n_atom

        atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs = \
            ConfGenModel.update_graph(compound_encoder_config=self.compound_encoder_config, data_list=new_data)

        node_repr, edge_repr, graph_repr, node_hidden_list = self.decoder_gnn(
            atom_bond_graphs, bond_angle_graphs, angle_dihedral_graphs, z=z
        )
        # node_repr = self.decoder_norm(node_repr)
        for i, layer in enumerate(self.decoder_pos):
            delta_pos = self.decoder_pos[i](node_repr)
            cur_pos, _ = ConfGenModel.move2origin(cur_pos + delta_pos, batch["batch"])
            # cur_pos = cur_pos + delta_pos
            pos_list.append(cur_pos)

        return extra_output, pos_list

    @staticmethod
    def update_graph(compound_encoder_config, data_list):
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

            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            angle_dihedral_graph_list.append(adi_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        angle_dihedral_graph = pgl.Graph.batch(angle_dihedral_graph_list)

        # TODO: reshape due to pgl limitations on the shape
        ConfGenModel._flat_shapes(atom_bond_graph.node_feat)
        ConfGenModel._flat_shapes(atom_bond_graph.edge_feat)
        ConfGenModel._flat_shapes(bond_angle_graph.node_feat)
        ConfGenModel._flat_shapes(bond_angle_graph.edge_feat)
        ConfGenModel._flat_shapes(angle_dihedral_graph.node_feat)
        ConfGenModel._flat_shapes(angle_dihedral_graph.edge_feat)

        return atom_bond_graph.tensor(), bond_angle_graph.tensor(), angle_dihedral_graph.tensor()

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
