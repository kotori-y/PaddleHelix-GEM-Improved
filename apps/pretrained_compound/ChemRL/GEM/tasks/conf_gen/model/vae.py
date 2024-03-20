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
            layer_num=model_config['prior_head_layer_num'],
            in_size=model.embed_dim,
            hidden_size=model_config['prior_head_hidden_dim'],
            out_size=model.embed_dim * 2,
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
            layer_num=model_config['encoder_head_layer_num'],
            in_size=model.embed_dim,
            hidden_size=model_config['encoder_head_hidden_dim'],
            out_size=model.embed_dim * 2,
            act=model_config['act'],
            dropout_rate=model_config['dropout_rate']
        )

    def forward(self, encoder_graph):
        _, _, graph_repr = self.model(**encoder_graph)
        latent = self.encoder_head(graph_repr)
        mu_q, sigma_q = paddle.chunk(latent, chunks=2, axis=-1)
        return mu_q, sigma_q


class ConfDecoder(nn.Layer):
    def __init__(self, model: GNNModel, model_config):
        super().__init__()
        self.model = model

        self.layer_num = model_config['decoder_head_layer_num']
        self.hidden_dim = model_config['decoder_head_hidden_dim']
        self.act = model_config['act']
        self.dropout_rate = model_config['dropout_rate']
        self.pretrain_tasks = model_config['pretrain_tasks']

        # bond angle with regression
        if 'Bar' in self.pretrain_tasks:
            self.Bar_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim * 3,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.Bar_loss = nn.SmoothL1Loss()
        # dihedral angle with regression
        if 'Dir' in self.pretrain_tasks:
            self.Dir_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim * 4,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.Dir_loss = nn.SmoothL1Loss()
        # bond length with regression
        if 'Blr' in self.pretrain_tasks:
            self.Blr_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.graph_dim * 2,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.Blr_loss = nn.SmoothL1Loss()
        # atom distance with classification
        if 'Adc' in self.pretrain_tasks:
            self.Adc_vocab = model_config['Adc_vocab']
            self.Adc_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim * 2,
                hidden_size=self.hidden_dim,
                out_size=self.Adc_vocab + 3,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.Adc_loss = nn.CrossEntropyLoss()

        # cm5 charge prediction
        if 'cm5' in self.pretrain_tasks:
            self.cm5_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.cm5_loss = nn.SmoothL1Loss()
        # espc charge prediction
        if 'espc' in self.pretrain_tasks:
            self.espc_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.espc_loss = nn.SmoothL1Loss()
        # hirshfeld charge prediction
        if 'hirshfeld' in self.pretrain_tasks:
            self.hirshfeld_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.hirshfeld_loss = nn.SmoothL1Loss()
        # npa charge prediction
        if 'npa' in self.pretrain_tasks:
            self.npa_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.npa_loss = nn.SmoothL1Loss()
        # bond order with regression
        if 'wiberg' in self.pretrain_tasks:
            self.bo_mlp = MLP(
                layer_num=self.layer_num,
                in_size=model.embed_dim * 2,
                hidden_size=self.hidden_dim,
                out_size=1,
                act=self.act,
                dropout_rate=self.dropout_rate
            )
            # self.bo_loss = nn.SmoothL1Loss()

        self.latent_emb = nn.Linear(model.embed_dim, model.embed_dim)

    def _get_Bar(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, feed_dict['Ba_node_k'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        return self.Bar_mlp(node_ijk_repr)

    def _get_Dir(self, feed_dict, node_repr):
        node_a_repr = paddle.gather(node_repr, feed_dict['Adi_node_a'])
        node_b_repr = paddle.gather(node_repr, feed_dict['Adi_node_b'])
        node_c_repr = paddle.gather(node_repr, feed_dict['Adi_node_c'])
        node_d_repr = paddle.gather(node_repr, feed_dict['Adi_node_d'])
        node_abcd_repr = paddle.concat([node_a_repr, node_b_repr, node_c_repr, node_d_repr], 1)
        return self.Dir_mlp(node_abcd_repr)

    def _get_Blr(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        return self.Blr_mlp(node_ij_repr)

    def _get_Adc(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        return self.Adc_mlp(node_ij_repr)

    def _get_cm5(self, node_repr):
        return self.cm5_mlp(node_repr)

    def _get_espc(self, node_repr):
        return self.espc_mlp(node_repr)

    def _get_hirshfeld(self, node_repr):
        return self.hirshfeld_mlp(node_repr)

    def _get_npa(self, node_repr):
        return self.npa_mlp(node_repr)

    def _get_bo(self, feed_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, feed_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, feed_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        return self.bo_mlp(node_ij_repr)

    def forward(self, decoder_graph, decoder_feed, latent, batch):
        node_repr, _, _ = self.model(**decoder_graph)

        latent = self.latent_emb(latent)
        latent = paddle.to_tensor(np.repeat(latent.numpy(), batch["num_nodes"], axis=0))

        node_repr = paddle.add(node_repr, latent)

        ans = {}

        if 'Bar' in self.pretrain_tasks:
            ans['Bar'] = self._get_Bar(decoder_feed, node_repr)
        if 'Dir' in self.pretrain_tasks:
            ans['Dir'] = self._get_Dir(decoder_feed, node_repr)
        if 'Blr' in self.pretrain_tasks:
            ans['Blr'] = self._get_Blr(decoder_feed, node_repr)
        if 'Adc' in self.pretrain_tasks:
            ans['Adc'] = self._get_Adc(decoder_feed, node_repr)
        if 'cm5' in self.pretrain_tasks:
            ans['cm5'] = self._get_cm5(node_repr)
        if 'espc' in self.pretrain_tasks:
            ans['espc'] = self._get_espc(node_repr)
        if 'hirshfeld' in self.pretrain_tasks:
            ans['hirshfeld'] = self._get_hirshfeld(node_repr)
        if 'npa' in self.pretrain_tasks:
            ans['npa'] = self._get_npa(node_repr)
        if 'wiberg' in self.pretrain_tasks:
            ans['wiberg'] = self._get_bo(decoder_feed, node_repr)

        return ans


class VAE(nn.Layer):
    def __init__(self, prior: GNNModel, encoder: GeoGNNModel, decoder: GNNModel, head_config, vae_beta=1.0):
        super().__init__()

        assert prior.embed_dim == encoder.embed_dim
        head_config['embed_dim'] = prior.embed_dim
        self.vae_beta = vae_beta

        # p(z | G)
        self.prior = ConfPrior(prior, head_config)

        # q(z | G', H', I')
        self.encoder = ConfEncoder(encoder, head_config)

        # p(G, H, I | G', z)
        self.decoder = ConfDecoder(decoder, head_config)

        self.pretrain_tasks = head_config['pretrain_tasks']
        self.Adc_vocab = head_config['Adc_vocab']

        # bond angle with regression
        if 'Bar' in self.pretrain_tasks:
            self.Bar_loss = nn.SmoothL1Loss()
        # dihedral angle with regression
        if 'Dir' in self.pretrain_tasks:
            self.Dir_loss = nn.SmoothL1Loss()
        # bond length with regression
        if 'Blr' in self.pretrain_tasks:
            self.Blr_loss = nn.SmoothL1Loss()
        # atom distance with classification
        if 'Adc' in self.pretrain_tasks:
            self.Adc_loss = nn.CrossEntropyLoss()

        # cm5 charge prediction
        if 'cm5' in self.pretrain_tasks:
            self.cm5_loss = nn.SmoothL1Loss()
        # espc charge prediction
        if 'espc' in self.pretrain_tasks:
            self.espc_loss = nn.SmoothL1Loss()
        # hirshfeld charge prediction
        if 'hirshfeld' in self.pretrain_tasks:
            self.hirshfeld_loss = nn.SmoothL1Loss()
        # npa charge prediction
        if 'npa' in self.pretrain_tasks:
            self.npa_loss = nn.SmoothL1Loss()
        # bond order with regression
        if 'wiberg' in self.pretrain_tasks:
            self.bo_loss = nn.SmoothL1Loss()

    def forward(self, prior_graph, encoder_graph, decoder_graph, decoder_feed, batch):
        mu_p, sigma_p = self.prior(prior_graph)
        mu_q, sigma_q = self.encoder(encoder_graph)

        latent = self.reparameterize_gaussian(mu_q, sigma_q)

        geometry_dict = self.decoder(decoder_graph, decoder_feed, latent, batch)
        loss, sub_losses = self.compute_geometry_loss(geometry_dict, decoder_feed)
        # KL Distance
        loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)

        loss += (loss_kl * self.vae_beta)
        sub_losses['kl_loss'] = loss_kl
        sub_losses['loss'] = loss

        for name in sub_losses:
            sub_losses[name] = sub_losses[name].numpy().mean()

        return loss, sub_losses

    @staticmethod
    def compute_vae_kl(mu_q, logvar_q, mu_prior, logvar_prior):
        mu1 = mu_q
        std1 = paddle.exp(0.5 * logvar_q)
        mu2 = mu_prior
        std2 = paddle.exp(0.5 * logvar_prior)
        kl = - 0.5 + paddle.log(std2 / (std1 + 1e-8) + 1e-8) + \
             ((paddle.pow(std1, 2) + paddle.pow(mu1 - mu2, 2)) / (2 * paddle.pow(std2, 2)))

        bs = kl.shape[0]
        kl = kl.sum() / bs

        return kl

    def compute_geometry_loss(self, geometry_dict, feed_dict):
        sub_losses = {}

        if 'Bar' in self.pretrain_tasks:
            sub_losses['Bar_loss'] = self.Bar_loss(geometry_dict['Bar'], feed_dict['Ba_bond_angle'])
        if 'Dir' in self.pretrain_tasks:
            sub_losses['Dir_loss'] = self.Dir_loss(geometry_dict['Dir'], feed_dict['Adi_angle_dihedral'])
        if 'Blr' in self.pretrain_tasks:
            sub_losses['Blr_loss'] = self.Blr_loss(geometry_dict['Blr'], feed_dict['Bl_bond_length'])
        if 'Adc' in self.pretrain_tasks:
            logits = geometry_dict['Adc']
            atom_dist = paddle.clip(feed_dict['Ad_atom_dist'], 0.0, 20.0)
            atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Adc_vocab, 'int64')
            sub_losses['Adc_loss'] = self.Adc_loss(logits, atom_dist_id)
        if 'cm5' in self.pretrain_tasks:
            sub_losses['cm5_loss'] = self.cm5_loss(geometry_dict['cm5'], feed_dict['atom_cm5'].unsqueeze(1))
        if 'espc' in self.pretrain_tasks:
            sub_losses['espc_loss'] = self.espc_loss(geometry_dict['espc'], feed_dict['atom_espc'].unsqueeze(1))
        if 'hirshfeld' in self.pretrain_tasks:
            sub_losses['hirshfeld_loss'] = self.hirshfeld_loss(geometry_dict['hirshfeld'], feed_dict['atom_hirshfeld'].unsqueeze(1))
        if 'npa' in self.decoder.pretrain_tasks:
            sub_losses['npa_loss'] = self.npa_loss(geometry_dict['npa'], feed_dict['atom_npa'].unsqueeze(1))
        if 'wiberg' in self.decoder.pretrain_tasks:
            sub_losses['bo_loss'] = self.bo_loss(geometry_dict['wiberg'], feed_dict['bo_bond_order'])

        loss = 0
        for name in sub_losses:
            loss += sub_losses[name]

        return loss, sub_losses

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = paddle.exp(0.5 * logvar)
        eps = paddle.randn(std.shape)
        return mean + std * eps
