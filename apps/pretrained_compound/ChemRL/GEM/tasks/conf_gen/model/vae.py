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
    def __init__(self, n_layers, prior_config, head_config):
        super().__init__()

        assert n_layers % 3 == 0

        self.layers = nn.LayerList(
            [
                ConfPriorLayer(prior_config, head_config) for _ in range(n_layers)
            ]
        )

    def forward(self, prior_graph, prior_feed, prior_batch):
        prior_positions_list = []
        # prior_bond_length_list = []
        # prior_bond_angle_list = []
        # prior_dihedral_angle_list = []

        new_positions = prior_batch["positions"]

        for i, layer in enumerate(self.layers):
            delta_positions = layer(prior_graph)

            flag = i % 3

            prior_graph, new_positions, new_target_values = updated_graph(
                graph=prior_graph,
                feed_dict=prior_feed,
                now_positions=new_positions,
                delta_positions=delta_positions,
                update_target=TARGET_MAPPING[flag]
            )

            prior_positions_list.append(new_positions)

            # if flag == 0:
            #     prior_bond_length_list.append(new_target_values)
            # elif flag == 1:
            #     prior_bond_angle_list.append(new_target_values)
            # else:
            #     prior_dihedral_angle_list.append(new_target_values)

        # return prior_positions_list, prior_bond_length_list, prior_bond_angle_list, prior_dihedral_angle_list
        return prior_positions_list


class ConfEncoder(nn.Layer):
    def __init__(self, n_layers, encoder_config, head_config):
        super().__init__()

        assert n_layers % 3 == 0

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

        new_positions = encoder_batch["positions"]

        for i, layer in enumerate(self.layers):
            delta_positions, graph_repr = layer(encoder_graph)

            flag = i % 3

            encoder_graph, new_positions, _, = updated_graph(
                graph=encoder_graph,
                feed_dict=prior_feed,
                now_positions=new_positions,
                delta_positions=delta_positions,
                update_target=TARGET_MAPPING[flag]
            )

        latent = self.encoder_head(graph_repr)
        mu_q, sigma_q = paddle.chunk(latent, chunks=2, axis=-1)
        return mu_q, sigma_q


class ConfDecoder(nn.Layer):
    def __init__(self, n_layers, decoder_config, head_config):
        super().__init__()

        assert n_layers % 3 == 0

        self.layers = nn.LayerList(
            [
                ConfDecoderLayer(decoder_config, head_config) for _ in range(n_layers)
            ]
        )

    def forward(self, decoder_graph, prior_feed, decoder_batch, latent):
        decoder_positions_list = []
        decoder_bond_length_list = []
        decoder_bond_angle_list = []
        decoder_dihedral_angle_list = []

        new_positions = decoder_batch["positions"]

        for i, layer in enumerate(self.layers):
            delta_positions = layer(decoder_graph, decoder_batch, latent)

            flag = i % 3

            prior_graph, new_positions, new_target_values = updated_graph(
                graph=decoder_graph,
                feed_dict=prior_feed,
                now_positions=new_positions,
                delta_positions=delta_positions,
                update_target=TARGET_MAPPING[flag]
            )

            decoder_positions_list.append(new_positions)

            if flag == 0:
                decoder_bond_length_list.append(new_target_values)
            elif flag == 1:
                decoder_bond_angle_list.append(new_target_values)
            else:
                decoder_dihedral_angle_list.append(new_target_values)

        return decoder_positions_list, decoder_bond_length_list, decoder_bond_angle_list, decoder_dihedral_angle_list


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
        self.embed_dim = prior_config['embed_dim']
        self.n_layers = n_layers

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
        self.bond_angle_loss = nn.SmoothL1Loss()
        self.dihedral_angle_loss = nn.SmoothL1Loss()

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
            extra_output["gt_bond_length"] = encoder_feed["Bl_bond_length"]
            extra_output["gt_bond_angle"] = encoder_feed["Ba_bond_angle"]
            extra_output["gt_dihedral_angle"] = encoder_feed["Adi_angle_dihedral"]

        # prior
        prior_positions_list = self.prior(prior_graph, prior_feed, prior_batch)

        extra_output["prior_positions_list"] = prior_positions_list
        extra_output["batch_dict"] = prior_batch
        # prior

        # encoder
        if not sample:
            mu, sigma = self.encoder(encoder_graph, prior_feed, encoder_batch)
            latent = self.reparameterize_gaussian(mu, sigma)
            extra_output["latent_mean"] = mu
            extra_output["latent_logstd"] = sigma
        else:
            latent = paddle.randn((prior_batch['num_nodes'].shape[0], self.embed_dim))
        # encoder

        # decoder
        decoder_graph, _, _, = updated_graph(
            graph=decoder_graph,
            feed_dict=prior_feed,
            now_positions=prior_positions_list[-1],
            delta_positions=0,
            update_target='bond_length'
        )

        decoder_positions_list, decoder_bond_length_list, decoder_bond_angle_list, decoder_dihedral_angle_list = \
            self.decoder(decoder_graph, prior_feed, decoder_batch, latent)

        extra_output["decoder_bond_length_list"] = decoder_bond_length_list
        extra_output["decoder_bond_angle_list"] = decoder_bond_angle_list
        extra_output["decoder_dihedral_angle_list"] = decoder_dihedral_angle_list
        # decoder

        if compute_loss:
            loss, loss_dict = self.compute_loss(decoder_positions_list, encoder_batch, extra_output)
            return loss, loss_dict, decoder_positions_list

        return decoder_positions_list

    def compute_loss(self, decoder_positions_list, encoder_batch, extra_output):
        # kld loss
        mean = extra_output["latent_mean"]
        log_std = extra_output["latent_logstd"]
        loss_kld, loss_dict_kld = self.compute_kld_loss(mean, log_std)

        # position loss
        loss_position, loss_dict_position = self.compute_positions_loss(
            gt_positions=extra_output["gt_positions"],
            prior_positions_list=extra_output["prior_positions_list"],
            decoder_positions_list=decoder_positions_list,
            encoder_batch=encoder_batch,
            weight=5
        )

        # geometry_loss
        loss_geometry, loss_dict_geometry = self.compute_geometry_loss(extra_output)

        loss = loss_kld + loss_position + loss_geometry
        # loss = loss_kld + loss_geometry
        loss_dict = {
            **loss_dict_kld,
            **loss_dict_position,
            **loss_dict_geometry,
            "loss": loss.numpy()[0]
        }

        return loss, loss_dict

    def compute_kld_loss(self, mean, log_std):
        kld = -0.5 * paddle.sum(1 + 2 * log_std - mean.pow(2) - paddle.exp(2 * log_std), axis=-1)
        kld = kld.mean()  # todo check this line

        return kld * self.vae_beta, {'loss_kld': kld.numpy()[0]}

    def compute_positions_loss(self, gt_positions, prior_positions_list, decoder_positions_list, encoder_batch, weight):
        pos_x = decoder_positions_list[-1]
        pos_y = gt_positions
        new_idx = self.update_iso(pos_y, pos_x, encoder_batch)

        # prior positions loss
        loss_prior_position, _ = alignment_loss(
            # pos, pos_list[i].index_select(0, new_idx), batch, clamp=args.clamp_dist
            gt_positions,
            paddle.index_select(prior_positions_list[-1], axis=0, index=new_idx),
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
        loss_decoder_position, _ = alignment_loss(
            gt_positions,
            paddle.index_select(decoder_positions_list[-1], axis=0, index=new_idx),
            batch=encoder_batch
        )
        # decoder positions loss

        loss_dict = {
            "loss_prior_position": loss_prior_position.numpy()[0],
            "loss_decoder_position": loss_decoder_position.numpy()[0],
        }
        loss = (loss_prior_position + loss_decoder_position * self.n_layers) * weight

        return loss, loss_dict

    def _compute_bond_length_loss(self, gt_bond_length, decoder_bond_length_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        for i, bond_length in enumerate(decoder_bond_length_list):
            loss_decoder = self.bond_length_loss(
                bond_length,
                gt_bond_length
            )
            loss += (loss_decoder * weight * (1 + (i * step)))
            loss_dict[f"loss_decoder_bond_length_{i}"] = loss_decoder.numpy()[0]

        return loss, loss_dict

    def _compute_bond_angle_loss(self, gt_bond_angle, decoder_bond_angle_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        for i, bond_angle in enumerate(decoder_bond_angle_list):
            loss_decoder = self.bond_angle_loss(
                bond_angle,
                gt_bond_angle
            )
            loss += (loss_decoder * weight * (1 + (i * step)))
            loss_dict[f"loss_decoder_bond_angle_{i}"] = loss_decoder.numpy()[0]

        return loss, loss_dict

    def _compute_dihedral_angle_loss(self, gt_dihedral_angle, decoder_dihedral_angle_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        for i, dihedral_angle in enumerate(decoder_dihedral_angle_list):
            loss_decoder = self.dihedral_angle_loss(
                dihedral_angle.unsqueeze(1),
                gt_dihedral_angle
            )
            loss += (loss_decoder * weight * (1 + (i * step)))
            loss_dict[f"loss_decoder_dihedral_angle_{i}"] = loss_decoder.numpy()[0]

        return loss, loss_dict

    def compute_geometry_loss(self, extra_output):
        loss_bond_length, loss_dict_bond_length = self._compute_bond_length_loss(
            gt_bond_length=extra_output["gt_bond_length"],
            decoder_bond_length_list=extra_output["decoder_bond_length_list"],
            weight=1,
            step=0.2
        )

        loss_bond_angle, loss_dict_bond_angle = self._compute_bond_angle_loss(
            gt_bond_angle=extra_output["gt_bond_angle"],
            decoder_bond_angle_list=extra_output["decoder_bond_angle_list"],
            weight=2,
            step=0.2
        )

        loss_dihedral_angle, loss_dict_dihedral_angle = self._compute_dihedral_angle_loss(
            gt_dihedral_angle=extra_output["gt_dihedral_angle"],
            decoder_dihedral_angle_list=extra_output["decoder_dihedral_angle_list"],
            weight=2,
            step=0.2
        )

        loss = loss_bond_length + loss_bond_angle + loss_dihedral_angle
        # loss = loss_bond_length + loss_bond_angle

        loss_dict = {
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
