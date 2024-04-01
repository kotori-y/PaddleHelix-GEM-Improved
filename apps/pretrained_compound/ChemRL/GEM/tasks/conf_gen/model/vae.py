import paddle
import paddle.nn as nn
# from rdkit.Chem import rdDepictor as DP
from rdkit import RDLogger

try:
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.loss.e3_loss import alignment_loss
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
except:
    from conf_gen.loss.e3_loss import alignment_loss
    from conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
from pahelix.model_zoo.gem_model import GNNModel
from pahelix.networks.basic_block import MLP

RDLogger.DisableLog('rdApp.*')


def get_bond_length(positions, feed_dict):
    position_i = paddle.gather(positions, feed_dict['Bl_node_i'])
    position_j = paddle.gather(positions, feed_dict['Bl_node_j'])

    bond_length = paddle.norm(position_i - position_j, p=2, axis=1).unsqueeze(1)
    return bond_length


def updated_graph(graph, feed_dict, batch_dict, delta_positions, new_positions=None):
    new_graph = graph.copy()

    if new_positions is None:
        new_positions = batch_dict["positions"] + delta_positions
    new_bond_length = get_bond_length(new_positions, feed_dict)

    atom_bond_graph = new_graph['atom_bond_graph']
    atom_bond_graph.edge_feat['bond_length'] = new_bond_length.squeeze(1)

    return new_graph, new_positions, new_bond_length


class ConfPrior(nn.Layer):
    def __init__(self, n_layers, prior_config, head_config):
        super().__init__()

        self.layers = nn.LayerList(
            [
                ConfPriorLayer(prior_config, head_config) for _ in range(n_layers)
            ]
        )

    def forward(self, prior_graph, prior_feed, prior_batch):
        prior_positions_list = []
        prior_bond_length_list = []

        for i, layer in enumerate(self.layers):
            delta_positions = layer(prior_graph)

            prior_graph, new_positions, new_bond_length = updated_graph(
                prior_graph, prior_feed, prior_batch, delta_positions=delta_positions
            )

            prior_positions_list.append(new_positions)
            prior_bond_length_list.append(new_bond_length)

        return prior_positions_list, prior_bond_length_list


class ConfEncoder(nn.Layer):
    def __init__(self, n_layers, encoder_config, head_config):
        super().__init__()

        self.layers = nn.LayerList(
            [
                ConfEncoderLayer(encoder_config, head_config) for _ in range(n_layers)
            ]
        )

        self.encoder_head = MLP(
            layer_num=head_config['encoder_head_layer_num'],
            in_size=encoder_config['embed_dim'],
            hidden_size=head_config['encoder_head_hidden_dim'],
            out_size=encoder_config['embed_dim'] * 2,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )

    def forward(self, encoder_graph, prior_feed, encoder_batch):
        assert len(self.layers) >= 1

        for i, layer in enumerate(self.layers):
            delta_positions, graph_repr = layer(encoder_graph)

            encoder_graph, _, _ = updated_graph(
                encoder_graph, prior_feed, encoder_batch, delta_positions=delta_positions
            )

        latent = self.encoder_head(graph_repr)
        mu_q, sigma_q = paddle.chunk(latent, chunks=2, axis=-1)
        return mu_q, sigma_q


class ConfDecoder(nn.Layer):
    def __init__(self, n_layers, decoder_config, head_config):
        super().__init__()

        self.layers = nn.LayerList(
            [
                ConfDecoderLayer(decoder_config, head_config) for _ in range(n_layers)
            ]
        )

    def forward(self, decoder_graph, prior_feed, decoder_batch, latent):
        decoder_positions_list = []
        decoder_bond_length_list = []

        for i, layer in enumerate(self.layers):
            delta_positions = layer(decoder_graph, decoder_batch, latent)

            prior_graph, new_positions, new_bond_length = updated_graph(
                decoder_graph, prior_feed, decoder_batch, delta_positions=delta_positions
            )

            decoder_positions_list.append(new_positions)
            decoder_bond_length_list.append(new_bond_length)

        return decoder_positions_list, decoder_bond_length_list


class VAE(nn.Layer):
    def __init__(
            self,
            prior_config,
            encoder_config,
            decoder_config,
            head_config,
            n_layers,
            vae_beta=1.0,
            aux_weight=1.0
    ):
        super().__init__()

        # assert prior.embed_dim == encoder.embed_dim
        # head_config['embed_dim'] = prior.embed_dim
        self.vae_beta = vae_beta
        self.aux_weight = aux_weight

        # p(z | G)
        self.prior = ConfPrior(n_layers, prior_config, head_config)

        # q(z | G', H', I')
        self.encoder = ConfEncoder(n_layers, encoder_config, head_config)

        # p(G, H, I | G', z)
        self.decoder = ConfDecoder(n_layers, decoder_config, head_config)

        # self.Bar_loss = nn.SmoothL1Loss()
        # self.Dir_loss = nn.SmoothL1Loss()
        self.bond_length_loss = nn.SmoothL1Loss()

    def forward(
            self,
            prior_graph, encoder_graph, decoder_graph,
            prior_feed, encoder_feed, decoder_feed,
            prior_batch, encoder_batch, decoder_batch,
            evaluate=False
    ):
        extra_output = {
            "n_mols": prior_batch["num_nodes"].shape[0]
        }

        # 'Ba_bond_angle', 'Adi_angle_dihedral'
        if not evaluate:
            extra_output["gt_positions"] = encoder_batch["positions"]
            extra_output["gt_bond_length"] = encoder_feed["Bl_bond_length"]

        prior_positions_list, prior_bond_length_list = self.prior(prior_graph, prior_feed, prior_batch)
        extra_output["prior_positions_list"] = prior_positions_list
        extra_output["prior_bond_length_list"] = prior_bond_length_list
        extra_output["batch_dict"] = prior_batch

        mu, sigma = self.encoder(encoder_graph, prior_feed, encoder_batch)
        latent = self.reparameterize_gaussian(mu, sigma)
        extra_output["latent_mean"] = mu
        extra_output["latent_logstd"] = sigma

        decoder_graph, _, _ = updated_graph(
            decoder_graph, prior_feed, decoder_batch,
            delta_positions=None, new_positions=prior_positions_list[-1]
        )
        decoder_positions_list, decoder_bond_length_list = self.decoder(
            decoder_graph, prior_feed, decoder_batch, latent
        )
        extra_output["decoder_bond_length_list"] = decoder_bond_length_list

        loss, loss_dict = self.compute_loss(decoder_positions_list, extra_output)

        return loss, loss_dict, decoder_positions_list

    def compute_loss(self, decoder_positions_list, extra_output):
        loss_dict = {}
        loss = 0

        # kld loss
        mean = extra_output["latent_mean"]
        log_std = extra_output["latent_logstd"]
        kld = -0.5 * paddle.sum(1 + 2 * log_std - mean.pow(2) - paddle.exp(2 * log_std), axis=-1)
        kld = kld.mean()  # todo check this line
        loss = loss + kld * self.vae_beta
        loss_dict["loss_kld"] = kld.numpy()[0]

        # new_idx = self.update_iso(pos, pos_list[-1], batch)

        # prior positions loss
        loss_tmp, _ = alignment_loss(
            extra_output["gt_positions"], extra_output["prior_positions_list"][-1], extra_output["batch_dict"]
        )
        loss += loss_tmp * 2  # todo
        loss_dict["loss_prior_position"] = loss_tmp.numpy()[0]

        # decoder positions loss
        for i, position in enumerate(decoder_positions_list):
            loss_tmp, _ = alignment_loss(extra_output["gt_positions"], position, extra_output["batch_dict"])
            loss += loss_tmp * (1.0 if i == 0 else self.aux_weight)
            loss_dict[f"loss_pos_{i}"] = loss_tmp.numpy()[0]

        # geometry loss
        _loss, _loss_dict = self.compute_geometry_loss(extra_output)
        loss += _loss
        loss_dict = {**loss_dict, **_loss_dict}

        # for task in ['bond_length', 'bond_angle']:
        #     positions = self.decoder(decoder_graph, decoder_feed, latent, batch)
        #     sub_losses[f"{task}_loss"] = self.compute_geometry_loss(positions, decoder_feed, pretrain_task=task)
        #     self.update_decoder_graph(decoder_graph, positions)
        # # KL Distance
        # loss_kl = self.compute_vae_kl(mu_q, sigma_q, mu_p, sigma_p)
        #
        # loss += (loss_kl * self.vae_beta)
        # sub_losses['kl_loss'] = loss_kl
        # sub_losses['loss'] = loss
        #
        # for name in sub_losses:
        #     sub_losses[name] = sub_losses[name].numpy().mean()
        return loss, loss_dict

    def compute_geometry_loss(self, extra_output):
        loss = 0
        loss_dict = {}

        blr_loss = nn.SmoothL1Loss()

        loss_tmp = blr_loss(
            extra_output["prior_bond_length_list"][-1], extra_output["gt_bond_length"]
        )
        loss += loss_tmp * 20  # todo
        loss_dict[f"loss_prior_bond_length"] = loss_tmp.numpy()[0]

        for i, bond_length in enumerate(extra_output["decoder_bond_length_list"]):
            loss_tmp = blr_loss(bond_length, extra_output["gt_bond_length"])
            weight = 10  # todo

            loss += loss_tmp * weight
            loss_dict[f"loss_decoder_bond_length_{i}"] = loss_tmp.numpy()[0]

        return loss, loss_dict

        # if 'Bar' in pretrain_tasks:
        #     sub_losses['Bar_loss'] = self.Bar_loss(geometry_dict['Bar'], feed_dict['Ba_bond_angle'])
        # if 'Dir' in self.pretrain_tasks:
        #     sub_losses['Dir_loss'] = self.Dir_loss(geometry_dict['Dir'], feed_dict['Adi_angle_dihedral'])

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = paddle.exp(0.5 * logvar)
        eps = paddle.randn(std.shape)
        return mean + std * eps

    @staticmethod
    def update_iso(pos_y, pos_x, batch):
        with paddle.no_grad():
            pre_nodes = 0
            num_nodes = batch.n_nodes
            isomorphisms = batch.isomorphisms
            new_idx_x = []
            for i in range(batch.num_graphs):
                cur_num_nodes = num_nodes[i]
                current_isomorphisms = [
                    torch.LongTensor(iso).to(pos_x.device) for iso in isomorphisms[i]
                ]
                if len(current_isomorphisms) == 1:
                    new_idx_x.append(current_isomorphisms[0] + pre_nodes)
                else:
                    pos_y_i = pos_y[pre_nodes: pre_nodes + cur_num_nodes]
                    pos_x_i = pos_x[pre_nodes: pre_nodes + cur_num_nodes]
                    pos_y_mean = torch.mean(pos_y_i, dim=0, keepdim=True)
                    pos_x_mean = torch.mean(pos_x_i, dim=0, keepdim=True)
                    pos_x_list = []

                    for iso in current_isomorphisms:
                        pos_x_list.append(torch.index_select(pos_x_i, 0, iso))
                    total_iso = len(pos_x_list)
                    pos_y_i = pos_y_i.repeat(total_iso, 1)
                    pos_x_i = torch.cat(pos_x_list, dim=0)
                    min_idx = GNN.alignment_loss_iso_onegraph(
                        pos_y_i,
                        pos_x_i,
                        pos_y_mean,
                        pos_x_mean,
                        num_nodes=cur_num_nodes,
                        total_iso=total_iso,
                    )
                    new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
                pre_nodes += cur_num_nodes

            return torch.cat(new_idx_x, dim=0)
