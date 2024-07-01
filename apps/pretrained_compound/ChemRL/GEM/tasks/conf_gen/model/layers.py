import numpy as np
import paddle
from paddle import nn as nn

from pahelix.model_zoo.gem_model import GNNModel, GeoGNNModel
from pahelix.networks.basic_block import MLP, MLPwoLastAct


class ConfPriorLayer(nn.Layer):
    def __init__(self, prior_config, head_config, compound_encoder):
        super().__init__()

        self.compound_encoder = compound_encoder
        self.pos_layer = MLP(
                layer_num=head_config['prior_head_layer_num'],
                in_size=prior_config['embed_dim'],
                hidden_size=head_config['prior_head_hidden_dim'],
                out_size=3,
                act=head_config['act'],
                dropout_rate=head_config['dropout_rate']
            )

    def forward(self, prior_graph):
        with paddle.no_grad():
            node_repr, _, graph_repr = self.compound_encoder(**prior_graph)

        positions = self.pos_layer(node_repr)
        return positions, graph_repr


class ConfEncoderLayer(nn.Layer):
    def __init__(self, encoder_config, head_config, compound_encoder):
        super().__init__()

        self.compound_encoder = compound_encoder
        # self.pos_layer = MLP(
        #     layer_num=head_config['prior_head_layer_num'],
        #     in_size=encoder_config['embed_dim'],
        #     hidden_size=head_config['prior_head_hidden_dim'],
        #     out_size=3,
        #     act=head_config['act'],
        #     dropout_rate=head_config['dropout_rate']
        # )

    def forward(self, encoder_graph):
        with paddle.no_grad():
            _, _, graph_repr = self.compound_encoder(**encoder_graph)

        return graph_repr
        # positions = self.pos_layer(node_repr)

        # return positions, graph_repr


class ConfDecoderLayer(nn.Layer):
    def __init__(self, decoder_config, head_config, compound_encoder):
        super().__init__()

        self.compound_encoder = compound_encoder
        self.pos_layer = MLP(
            layer_num=head_config['decoder_head_layer_num'],
            in_size=decoder_config['embed_dim'],
            hidden_size=head_config['decoder_head_hidden_dim'],
            out_size=3,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )
        # self.latent_emb = nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim'])
        self.norm_layer = nn.LayerNorm(decoder_config['embed_dim'])
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, decoder_graph, decoder_batch=None, latent=None):
        with paddle.no_grad():
            node_repr, _, _ = self.compound_encoder(**decoder_graph)

        if decoder_batch is not None and latent is not None:
            # repeat_times not available in current version of paddle in development env
            tmp = [[latent[i]] * int(decoder_batch["num_nodes"][i]) for i in range(len(decoder_batch["num_nodes"]))]
            latent = paddle.stack(sum(tmp, []))

            node_repr = self.norm_layer(latent + self.drop_layer(node_repr))

        positions = self.pos_layer(node_repr)
        return positions
