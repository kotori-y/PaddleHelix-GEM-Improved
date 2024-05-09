import numpy as np
import paddle
from paddle import nn as nn

from pahelix.model_zoo.gem_model import GNNModel, GeoGNNModel
from pahelix.networks.basic_block import MLP, MLPwoLastAct


class ConfPriorLayer(nn.Layer):
    def __init__(self, prior_config, head_config):
        super().__init__()
        self.encode_model = GeoGNNModel(prior_config)
        self.pos_layer = MLP(
                layer_num=head_config['prior_head_layer_num'],
                in_size=prior_config['embed_dim'],
                hidden_size=head_config['prior_head_hidden_dim'],
                out_size=3,
                act=head_config['act'],
                dropout_rate=head_config['dropout_rate']
            )

    def forward(self, prior_graph):
        node_repr, _, _ = self.encode_model(**prior_graph)
        return self.pos_layer(node_repr)


class ConfEncoderLayer(nn.Layer):
    def __init__(self, encoder_config, head_config):
        super().__init__()
        self.encode_model = GeoGNNModel(encoder_config)
        self.pos_layer = MLP(
            layer_num=head_config['prior_head_layer_num'],
            in_size=encoder_config['embed_dim'],
            hidden_size=head_config['prior_head_hidden_dim'],
            out_size=3,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )

    def forward(self, encoder_graph):
        node_repr, _, graph_repr = self.encode_model(**encoder_graph)
        positions = self.pos_layer(node_repr)

        return positions, graph_repr


class ConfDecoderLayer(nn.Layer):
    def __init__(self, decoder_config, head_config):
        super().__init__()
        self.decode_model = GeoGNNModel(decoder_config)
        self.pos_layer = MLP(
            layer_num=head_config['prior_head_layer_num'],
            in_size=decoder_config['embed_dim'],
            hidden_size=head_config['prior_head_hidden_dim'],
            out_size=3,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )
        self.latent_emb = nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim'])

    def forward(self, decoder_graph, decoder_batch, latent):
        node_repr, _, _ = self.decode_model(**decoder_graph)

        latent = self.latent_emb(latent)
        latent = paddle.to_tensor(np.repeat(latent.numpy(), decoder_batch["num_nodes"], axis=0))

        node_repr = paddle.add(node_repr, latent)
        positions = self.pos_layer(node_repr)
        return positions
