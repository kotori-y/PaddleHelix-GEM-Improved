import copy

import paddle
import paddle.nn as nn
# from rdkit.Chem import rdDepictor as DP
from rdkit import RDLogger


try:
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.loss.e3_loss import alignment_loss, move2origin
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.update_graph import updated_graph
except:
    from conf_gen.loss.e3_loss import alignment_loss
    from conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
    from conf_gen.model.update_graph import updated_graph

from pahelix.networks.basic_block import MLP

RDLogger.DisableLog('rdApp.*')


TARGET_MAPPING = ['bond_length', 'bond_angle', 'dihedral_angle']


class ConfPrior(nn.Layer):
    def __init__(self, n_layers, prior_config, head_config, compound_encoder):
        super().__init__()

        self.prior_layer = ConfPriorLayer(prior_config, head_config, compound_encoder=compound_encoder)
        self.prior_head = MLP(
            layer_num=head_config['prior_head_layer_num'],
            in_size=prior_config['embed_dim'],
            hidden_size=head_config['prior_head_hidden_dim'],
            out_size=prior_config['embed_dim'] * 2,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )

    def forward(self, prior_graph, prior_feed, prior_batch):
        new_positions = prior_batch["positions"]

        delta_positions, graph_repr = self.prior_layer(prior_graph)
        prior_graph, prior_positions, _, _ = updated_graph(
            graph=prior_graph,
            feed_dict=prior_feed,
            now_positions=new_positions,
            delta_positions=delta_positions,
            update_target=TARGET_MAPPING[-1],
            batch=prior_batch['batch'],
            num_nodes=prior_batch['num_nodes'],
            move=True
        )

        latent = self.prior_head(graph_repr)
        mu_p, sigma_p = paddle.chunk(latent, chunks=2, axis=-1)

        return prior_graph, prior_positions, mu_p, sigma_p
        # return prior_positions_list


class ConfEncoder(nn.Layer):
    def __init__(self, n_layers, encoder_config, head_config, compound_encoder):
        super().__init__()

        self.encoder_layer = ConfEncoderLayer(encoder_config, head_config, compound_encoder=compound_encoder)
        self.encoder_head = MLP(
            layer_num=head_config['encoder_head_layer_num'],
            in_size=encoder_config['embed_dim'],
            hidden_size=head_config['encoder_head_hidden_dim'],
            out_size=encoder_config['embed_dim'] * 2,
            act=head_config['act'],
            dropout_rate=head_config['dropout_rate']
        )

    def forward(self, encoder_graph, prior_feed, encoder_batch):

        graph_repr = self.encoder_layer(encoder_graph)
        latent = self.encoder_head(graph_repr)
        mu_q, sigma_q = paddle.chunk(latent, chunks=2, axis=-1)

        return mu_q, sigma_q


class ConfDecoder(nn.Layer):
    def __init__(self, n_layers, decoder_config, head_config, compound_encoder):
        super().__init__()

        assert n_layers % 3 == 0

        self.layers = nn.LayerList(
            [
                ConfDecoderLayer(decoder_config, head_config, compound_encoder=compound_encoder) for _ in range(n_layers)
            ]
        )

    def forward(self, decoder_graph, prior_feed, decoder_batch, latent):
        decoder_positions_list = []

        # decoder_bond_length_list = []
        # decoder_bond_angle_list = []
        # decoder_dihedral_angle_list = []

        decoder_masked_bond_length_list = []
        decoder_masked_bond_angle_list = []
        decoder_masked_dihedral_angle_list = []

        # decoder_init_bond_length = 0
        # decoder_init_bond_angle = 0
        #
        # decoder_delta_bond_length = 0
        # decoder_delta_bond_angle = 0

        new_positions = decoder_batch["positions"]

        for i, layer in enumerate(self.layers):
            delta_positions = layer(decoder_graph, decoder_batch, latent)

            flag = i % 3

            decoder_graph, new_positions, new_target_values, extrac_info = updated_graph(
                graph=decoder_graph,
                feed_dict=prior_feed,
                now_positions=new_positions,
                delta_positions=delta_positions,
                update_target=TARGET_MAPPING[flag],
                batch=decoder_batch['batch'],
                num_nodes=decoder_batch['num_nodes'],
                move=True
            )

            decoder_positions_list.append(new_positions)

            if flag == 0:
                decoder_masked_bond_length_list.append(new_target_values)

            elif flag == 1:
                decoder_masked_bond_length_list.append(extrac_info[0])
                decoder_masked_bond_angle_list.append(new_target_values)

            else:
                decoder_masked_bond_length_list.append(extrac_info[0])
                decoder_masked_bond_angle_list.append(extrac_info[1])
                decoder_masked_dihedral_angle_list.append(new_target_values)

        # return decoder_positions_list, \
        #     decoder_bond_length_list, decoder_bond_angle_list, decoder_dihedral_angle_list, \
        #     decoder_masked_bond_length_list, decoder_masked_bond_angle_list, decoder_masked_dihedral_angle_list, \
        #     decoder_delta_bond_length, decoder_delta_bond_angle
        # return decoder_positions_list
        return decoder_positions_list, \
            decoder_masked_bond_length_list, decoder_masked_bond_angle_list, decoder_masked_dihedral_angle_list
        # return decoder_positions_list


class VAE(nn.Layer):
    def __init__(
            self,
            prior_config,
            encoder_config,
            decoder_config,
            head_config,
            n_layers,
            compound_encoder,
            vae_beta=1.0,
            aux_weight=1.0,
            isomorphism=True,
    ):
        super().__init__()

        # assert prior.embed_dim == encoder.embed_dim
        # head_config['embed_dim'] = prior.embed_dim
        self.embed_dim = prior_config['embed_dim']
        self.n_layers = n_layers

        self.vae_beta = vae_beta
        self.aux_weight = aux_weight

        # p(z | G)
        self.prior = ConfPrior(n_layers, prior_config, head_config, compound_encoder=compound_encoder)

        # q(z | G', H', I')
        self.encoder = ConfEncoder(n_layers, encoder_config, head_config, compound_encoder=compound_encoder)

        # p(G, H, I | G', z)
        self.decoder = ConfDecoder(n_layers, decoder_config, head_config, compound_encoder=compound_encoder)

        # self.Bar_loss = nn.SmoothL1Loss()
        # self.Dir_loss = nn.SmoothL1Loss()
        self.bond_length_loss = nn.SmoothL1Loss()
        self.bond_angle_loss = nn.SmoothL1Loss()
        self.dihedral_angle_loss = nn.SmoothL1Loss()

        self.isomorphism = isomorphism

    def forward(
            self,
            prior_graph, prior_feed, prior_batch,
            decoder_graph, decoder_feed, decoder_batch,
            encoder_graph=None, encoder_feed=None, encoder_batch=None,
            sample=False, compute_loss=True
    ):
        extra_output = {
            "n_mols": prior_batch["num_nodes"].shape[0]
        }

        if encoder_feed and encoder_batch:
            extra_output["gt_positions"] = encoder_batch["positions"]
            extra_output["gt_masked_bond_length"] = encoder_feed["masked_Bl_bond_length"]
            extra_output["gt_masked_bond_angle"] = encoder_feed["masked_Ba_bond_angle"]
            extra_output["gt_masked_dihedral_angle"] = encoder_feed["masked_Adi_angle_dihedral"]

        # encoder
        if not sample:
            mu_q, sigma_q = self.encoder(encoder_graph, prior_feed, encoder_batch)
            extra_output["latent_mean_q"] = mu_q
            extra_output["latent_logstd_q"] = sigma_q
        # encoder

        # prior
        new_prior_graph, prior_positions, mu_p, sigma_p = self.prior(prior_graph, prior_feed, prior_batch)
        extra_output["latent_mean_p"] = mu_p
        extra_output["latent_logstd_p"] = sigma_p
        # prior

        if not sample:
            latent = self.reparameterize_gaussian(extra_output["latent_mean_q"], extra_output["latent_logstd_q"])
        else:
            latent = self.reparameterize_gaussian(extra_output["latent_mean_p"], extra_output["latent_logstd_p"])

        # decoder
        # for i in range(3):
        decoder_graph = copy.deepcopy(new_prior_graph)

        decoder_positions_list, \
            decoder_masked_bond_length_list, decoder_masked_bond_angle_list, decoder_masked_dihedral_angle_list = \
            self.decoder(decoder_graph, prior_feed, decoder_batch, latent)

        # extra_output["decoder_bond_length_list"] = decoder_bond_length_list
        # extra_output["decoder_bond_angle_list"] = decoder_bond_angle_list
        # extra_output["decoder_dihedral_angle_list"] = decoder_dihedral_angle_list

        extra_output["decoder_masked_bond_length_list"] = decoder_masked_bond_length_list
        extra_output["decoder_masked_bond_angle_list"] = decoder_masked_bond_angle_list
        extra_output["decoder_masked_dihedral_angle_list"] = decoder_masked_dihedral_angle_list

        # extra_output["decoder_delta_bond_length"] = decoder_delta_bond_length
        # extra_output["decoder_delta_bond_angle"] = decoder_delta_bond_angle
        # decoder

        if compute_loss:
            loss, loss_dict = self.compute_loss(
                prior_positions=prior_positions,
                decoder_positions_list=decoder_positions_list,
                encoder_batch=encoder_batch,
                extra_output=extra_output)
            return loss, loss_dict, decoder_positions_list

        return decoder_positions_list

    def compute_loss(self, prior_positions, decoder_positions_list, encoder_batch, extra_output):
        # kld loss
        mean_q = extra_output["latent_mean_q"]
        log_std_q = extra_output["latent_logstd_q"]
        mean_p = extra_output["latent_mean_p"]
        log_std_p = extra_output["latent_logstd_p"]

        loss_kld, loss_dict_kld = self.compute_vae_kl(mean_q, log_std_q, mean_p, log_std_p)

        # geometry_loss
        loss_geometry, loss_dict_geometry = self.compute_geometry_loss(extra_output, weights=[0.2, 0.2, 0.2])

        # position loss
        loss_position, loss_dict_position = self.compute_positions_loss(
            gt_positions=extra_output["gt_positions"],
            decoder_positions_list=decoder_positions_list,
            prior_positions=prior_positions,
            encoder_batch=encoder_batch,
            weight=5
        )

        loss = loss_kld + loss_position + loss_geometry
        # loss = loss_kld + loss_geometry
        loss_dict = {
            **loss_dict_kld,
            **loss_dict_position,
            **loss_dict_geometry,

            "loss": loss.numpy()[0]
        }

        return loss, loss_dict

    def compute_vae_kl(self, mu_q, logvar_q, mu_prior, logvar_prior):
        mu1 = mu_q
        std1 = paddle.exp(0.5 * logvar_q)
        mu2 = mu_prior
        std2 = paddle.exp(0.5 * logvar_prior)
        kl = - 0.5 + paddle.log(std2 / (std1 + 1e-8) + 1e-8) + \
            ((paddle.pow(std1, 2) + paddle.pow(mu1 - mu2, 2)) / (2 * paddle.pow(std2, 2)))

        bs = kl.shape[0]
        kld = kl.sum() / bs

        return kld * self.vae_beta, {'loss_kld': kld.numpy()[0]}

    def compute_kld_loss(self, mean, log_std):
        kld = -0.5 * paddle.sum(1 + 2 * log_std - mean.pow(2) - paddle.exp(2 * log_std), axis=-1)
        kld = kld.mean()  # todo check this line

        return kld * self.vae_beta, {'loss_kld': kld.numpy()[0]}

    def compute_positions_loss(self, gt_positions, prior_positions, decoder_positions_list, encoder_batch, weight):
        pos_x = decoder_positions_list[-1]
        pos_y = gt_positions

        if self.isomorphism:
            new_idx = self.update_iso(pos_y, pos_x, encoder_batch)

        # prior positions loss
        pos_x = paddle.index_select(prior_positions, axis=0, index=new_idx) \
            if self.isomorphism else prior_positions
        loss_prior_position, _ = alignment_loss(
            # pos, pos_list[i].index_select(0, new_idx), batch, clamp=args.clamp_dist
            pos_y=gt_positions,
            pos_x=pos_x,
            batch=encoder_batch
        )
        # prior positions loss

        # decoder positions loss
        # for i, position in enumerate(decoder_positions_list):
        #     loss_tmp, _ = alignment_loss(
        #         extra_output["gt_positions"],
        #         paddle.index_select(position, axis=0, index=new_idx),
        #         extra_output["batch_dict"]
        #     )
        #     loss += loss_tmp * (1.0 if i == 0 else self.aux_weight)
        #     loss_dict[f"loss_decoder_position_{i}"] = loss_tmp.numpy()[0]

        # decoder positions loss
        loss_decoder_position = 0
        loss_decoder_dict = {}

        for i, decoder_positions in enumerate(decoder_positions_list):
            pos_x = paddle.index_select(decoder_positions, axis=0, index=new_idx) \
                if self.isomorphism else decoder_positions

            _loss_decoder_position, _ = alignment_loss(
                pos_y=gt_positions,
                pos_x=pos_x,
                batch=encoder_batch
            )

            loss_decoder_position += _loss_decoder_position
            loss_decoder_dict = {
                **loss_decoder_dict,
                f"loss_decoder_position_{i}": _loss_decoder_position.numpy()[0]
            }
        # decoder positions loss

        loss_dict = {
            "loss_prior_position": loss_prior_position.numpy()[0],
            **loss_decoder_dict
        }
        loss = (loss_decoder_position + loss_prior_position) * weight
        # loss = loss_decoder_position * self.n_layers * weight

        return loss, loss_dict

    def _compute_bond_length_loss(self, masked_gt_bond_length, masked_bond_length_list, prefix='decoder', weight=1):
        loss_masked_bond_length = 0

        def _compute_loss(gt_length, pred_length):
            zero_mask = gt_length == 0
            return self.bond_length_loss(
                pred_length[~zero_mask],
                gt_length[~zero_mask]
            )

        for masked_bond_length in masked_bond_length_list:
            loss_masked_bond_length += _compute_loss(masked_gt_bond_length, masked_bond_length)

        loss = loss_masked_bond_length * weight
        loss_dict = {
            f"loss_{prefix}_masked_bond_length": loss_masked_bond_length.numpy()[0],
        }

        return loss, loss_dict

    def _compute_bond_angle_loss(self, masked_gt_bond_angle, masked_bond_angle_list, prefix='decoder', weight=1):
        loss_masked_bond_angle = 0

        def _compute_loss(gt_angle, pred_angle):
            zero_mask = gt_angle <= 0.01
            return self.bond_angle_loss(
                pred_angle[~zero_mask],
                gt_angle[~zero_mask]
            )

        for masked_bond_angle in masked_bond_angle_list:
            loss_masked_bond_angle += _compute_loss(masked_gt_bond_angle, masked_bond_angle)

        loss = loss_masked_bond_angle * weight
        loss_dict = {
            f"loss_{prefix}_masked_bond_angle": loss_masked_bond_angle.numpy()[0],
        }

        return loss, loss_dict

    def _compute_dihedral_angle_loss(self, masked_gt_dihedral_angle, masked_dihedral_angle_list, prefix='decoder', weight=1):
        loss_masked_dihedral_angle = 0

        def _compute_loss(gt_angle, pred_angle):
            zero_mask = gt_angle <= 0.01
            return self.dihedral_angle_loss(
                pred_angle.unsqueeze(1)[~zero_mask],
                gt_angle[~zero_mask]
            )

        for masked_dihedral_angle in masked_dihedral_angle_list:
            loss_masked_dihedral_angle += _compute_loss(masked_gt_dihedral_angle, masked_dihedral_angle)

        loss = loss_masked_dihedral_angle * weight

        loss_dict = {
            f"loss_{prefix}_masked_dihedral_angle": loss_masked_dihedral_angle.numpy()[0]
        }

        return loss, loss_dict

    def compute_geometry_loss(self, extra_output, weights=None):
        if weights is None:
            weights = [1, 1, 1]
        loss = 0
        loss_dict = {}

        for prefix in ['decoder']:

            loss_bond_length, loss_dict_bond_length = self._compute_bond_length_loss(
                masked_gt_bond_length=extra_output["gt_masked_bond_length"],
                masked_bond_length_list=extra_output[f"{prefix}_masked_bond_length_list"],
                weight=weights[0],
                prefix=prefix
            )

            loss_bond_angle, loss_dict_bond_angle = self._compute_bond_angle_loss(
                masked_gt_bond_angle=extra_output["gt_masked_bond_angle"],
                masked_bond_angle_list=extra_output[f"{prefix}_masked_bond_angle_list"],
                weight=weights[1],
                prefix=prefix
            )

            loss_dihedral_angle, loss_dict_dihedral_angle = self._compute_dihedral_angle_loss(
                masked_gt_dihedral_angle=extra_output["gt_masked_dihedral_angle"],
                masked_dihedral_angle_list=extra_output[f"{prefix}_masked_dihedral_angle_list"],
                weight=weights[2],
                prefix=prefix
            )

            loss += (loss_bond_length + loss_bond_angle + loss_dihedral_angle)

            loss_dict = {
                **loss_dict,
                **loss_dict_bond_length,
                **loss_dict_bond_angle,
                **loss_dict_dihedral_angle
            }

        return loss, loss_dict

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = paddle.exp(0.5 * logvar)
        eps = paddle.randn(std.shape)
        return mean + std * eps

    @staticmethod
    def update_iso(pos_y, pos_x, batch):
        with paddle.no_grad():
            pre_nodes = 0
            pre_isomorphism_nodes = 0

            num_nodes = batch["num_nodes"]
            isomorphism = batch["isomorphism"]
            isomorphism_num = batch["isomorphism_num"]

            new_idx_x = []
            for i, cur_num_nodes in enumerate(num_nodes):

                current_isomorphisms = []
                for j in range(isomorphism_num[i]):
                    current_isomorphisms.append(isomorphism[pre_isomorphism_nodes: pre_isomorphism_nodes + cur_num_nodes])
                    pre_isomorphism_nodes += cur_num_nodes

                if len(current_isomorphisms) == 1:
                    new_idx_x.append(current_isomorphisms[0] + pre_nodes)
                else:
                    pos_y_i = pos_y[pre_nodes: pre_nodes + cur_num_nodes]
                    pos_x_i = pos_x[pre_nodes: pre_nodes + cur_num_nodes]
                    pos_y_mean = paddle.mean(pos_y_i, axis=0)
                    pos_x_mean = paddle.mean(pos_x_i, axis=0)
                    pos_x_list = []

                    for iso in current_isomorphisms:
                        pos_x_list.append(paddle.index_select(pos_x_i, axis=0, index=iso))

                    total_iso = len(pos_x_list)
                    pos_y_i = paddle.concat([pos_y_i for _ in range(total_iso)], axis=0)
                    pos_x_i = paddle.concat(pos_x_list, axis=0)

                    min_idx = VAE.alignment_loss_iso_onegraph(
                        pos_y_i,
                        pos_x_i,
                        pos_y_mean,
                        pos_x_mean,
                        num_nodes=cur_num_nodes,
                        total_iso=total_iso,
                    )
                    new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)

                pre_nodes += cur_num_nodes

            return paddle.concat(new_idx_x, axis=0)

    @staticmethod
    def alignment_loss_iso_onegraph(pos_y, pos_x, pos_y_mean, pos_x_mean, num_nodes, total_iso):
        with paddle.no_grad():
            total_nodes = pos_y.shape[0]
            y = pos_y - pos_y_mean
            x = pos_x - pos_x_mean
            a = y + x
            b = y - x
            a = a.reshape((-1, 1, 3))
            b = b.reshape((-1, 3, 1))
            tmp0 = paddle.concat(
                [
                    paddle.zeros(shape=(1, 1, 1), dtype=b.dtype).expand((total_nodes, -1, -1)),
                    -b.transpose((0, 2, 1))
                ], axis=-1
            )
            eye = paddle.eye(3).unsqueeze(0).expand((total_nodes, -1, -1))
            a = a.expand((-1, 3, -1))

            tmp1 = paddle.cross(eye, a, axis=-1)
            tmp1 = paddle.concat([b, tmp1], axis=-1)
            tmp = paddle.concat([tmp0, tmp1], axis=1)

            tmpb = paddle.bmm(tmp.transpose((0, 2, 1)), tmp).reshape((-1, num_nodes, 16))
            tmpb = paddle.mean(tmpb, axis=1).reshape((-1, 4, 4))

            w, v = paddle.linalg.eigh(tmpb)
            min_q = v[:, :, 0]
            rotation = VAE.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - paddle.einsum("kj,kij->ki", pos_x_mean.expand((total_iso, -1)), rotation)

            i, m, n = rotation.shape
            rotation = paddle.tile(rotation, repeat_times=(1, num_nodes, 1))
            rotation = paddle.reshape(rotation, [i * num_nodes, m, n])

            m, n = t.shape
            t = paddle.tile(t, repeat_times=(1, num_nodes))
            t = paddle.reshape(t, [m * num_nodes, n])

            pos_x = paddle.einsum("kj,kij->ki", pos_x, rotation) + t
            loss = (pos_y - pos_x).norm(axis=-1, keepdim=True).reshape((-1, num_nodes)).mean(-1)
            return paddle.argmin(loss)

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        q0 = quaternion[:, 0]
        q1 = quaternion[:, 1]
        q2 = quaternion[:, 2]
        q3 = quaternion[:, 3]

        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        return paddle.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=-1).reshape((-1, 3, 3))
