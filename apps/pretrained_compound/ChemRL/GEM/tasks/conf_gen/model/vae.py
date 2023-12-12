import copy

import numpy as np
import paddle
import paddle.nn as nn
import pgl

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.loss.e3_loss import compute_loss
from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel
from pahelix.networks.basic_block import MLP, MLPwoLastAct
from pahelix.utils.compound_tools import mol_to_geognn_graph_data, mol_to_geognn_graph_data_MMFF3d, Compound3DKit

from rdkit import Chem
from rdkit.Chem import rdDepictor as DP
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class ConfPrior(nn.Layer):
    def __init__(self, model: GNNModel, model_config):
        super().__init__()
        self.model = model
        self.prior_head = MLP(
            layer_num=model_config['layer_num'],
            in_size=model.graph_dim,
            hidden_size=model_config['hidden_dim'],
            out_size=model_config['embed_dim'] * 2,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate']
        )

    def forward(self, prior_graph):
        _, _, graph_repr = self.model(**prior_graph)
        latent = self.prior_head(graph_repr)
        mu_p, sigma_p = paddle.chunk(latent, chunks=2, axis=-1)
        return mu_p, sigma_p


class ConfEncoder(nn.Layer):
    def __init__(self, model: GeoGNNModel, model_config):
        super().__init__()
        self.model = model
        self.encoder_head = MLP(
            layer_num=model_config['layer_num'],
            in_size=model.graph_dim,
            hidden_size=model_config['hidden_dim'],
            out_size=model_config['embed_dim'] * 2,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate']
        )

    def forward(self, encoder_graph):
        _, _, graph_repr = self.model(**encoder_graph)
        latent = self.encoder_head(graph_repr)
        mu_q, sigma_q = paddle.chunk(latent, chunks=2, axis=-1)
        return mu_q, sigma_q


class ConfDecoder(nn.Layer):
    def __init__(self):
        ...


class VAE(nn.Layer):
    def __init__(self, prior: GNNModel, encoder: GeoGNNModel, head_config):
        super().__init__()

        # p(z | G)
        self.prior = ConfPrior(prior, head_config)

        # q(z | G', H', I')
        self.encoder = ConfEncoder(encoder, head_config)

    def forward(self, prior_graph, encoder_graph):
        mu_p, sigma_p = self.prior(prior_graph)
        mu_q, sigma_q = self.encoder(encoder_graph)

        # KL Distance
        loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)

    @staticmethod
    def compute_vae_kl(mu_q, logvar_q, mu_prior, logvar_prior):
        mu1 = mu_q
        std1 = paddle.exp(0.5 * logvar_q)
        mu2 = mu_prior
        std2 = paddle.exp(0.5 * logvar_prior)
        kl = - 0.5 + paddle.log(std2 / (std1 + 1e-8) + 1e-8) + \
             ((paddle.pow(std1, 2) + paddle.pow(mu1 - mu2, 2)) / (2 * paddle.pow(std2, 2)))

        return kl

