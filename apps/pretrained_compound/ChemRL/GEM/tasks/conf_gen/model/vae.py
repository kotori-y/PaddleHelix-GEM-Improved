import math
import numpy as np

import paddle
import paddle.nn as nn
# from rdkit.Chem import rdDepictor as DP
from rdkit import RDLogger

try:
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.loss.e3_loss import alignment_loss, move2origin
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
except:
    from conf_gen.loss.e3_loss import alignment_loss
    from conf_gen.model.layers import ConfPriorLayer, ConfEncoderLayer, \
        ConfDecoderLayer
from pahelix.model_zoo.gem_model import GNNModel
from pahelix.networks.basic_block import MLP

RDLogger.DisableLog('rdApp.*')


TARGET_MAPPING = ['bond_length', 'bond_angle', 'dihedral_angle']


def get_bond_length(positions, feed_dict):
    position_i = paddle.gather(positions, feed_dict['Bl_node_i'])
    position_j = paddle.gather(positions, feed_dict['Bl_node_j'])

    bond_length = paddle.norm(position_i - position_j, p=2, axis=1).unsqueeze(1)
    return bond_length


def get_bond_angle(positions, feed_dict):
    def _get_angle(vec1, vec2):
        norm1 = paddle.norm(vec1, p=2, axis=1)
        norm2 = paddle.norm(vec2, p=2, axis=1)

        mask = (norm1.unsqueeze(axis=1) == 0) | (norm2.unsqueeze(axis=1) == 0)

        vec1 = vec1 / (norm1.unsqueeze(axis=1) + 1e-5)  # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2.unsqueeze(axis=1) + 1e-5)

        angle = paddle.acos(paddle.dot(vec1, vec2))
        angle[mask] = 0

        return angle

    position_i = paddle.gather(positions, feed_dict['Ba_node_i'])
    position_j = paddle.gather(positions, feed_dict['Ba_node_j'])
    position_k = paddle.gather(positions, feed_dict['Ba_node_k'])

    bond_angle = _get_angle(position_j - position_i, position_j - position_k)
    return bond_angle


def get_dihedral_angle(positions, feed_dict):
    position_a = paddle.gather(positions, feed_dict['Adi_node_a'])
    position_b = paddle.gather(positions, feed_dict['Adi_node_b'])
    position_c = paddle.gather(positions, feed_dict['Adi_node_c'])
    position_d = paddle.gather(positions, feed_dict['Adi_node_d'])

    rAB = position_b - position_a
    rBC = position_c - position_b
    rCD = position_d - position_c

    nABC = paddle.cross(rAB, rBC)
    nABCSqLength = paddle.sum(nABC * nABC, axis=1)

    nBCD = paddle.cross(rBC, rCD)
    nBCDSqLength = paddle.sum(nBCD * nBCD, axis=1)

    m = paddle.cross(nABC, rBC)

    angles = -paddle.atan2(
        paddle.sum(m * nBCD, axis=1) / (paddle.sqrt(nBCDSqLength * paddle.sum(m * m, axis=1)) + 1e-4),
        paddle.sum(nABC * nBCD, axis=1) / (paddle.sqrt(nABCSqLength * nBCDSqLength) + 1e-4)
    )

    return angles


def updated_graph(graph, feed_dict, now_positions, delta_positions, batch, num_nodes, update_target):
    new_graph = graph.copy()

    new_positions = now_positions + delta_positions

    atom_bond_graph = new_graph['atom_bond_graph']
    bond_angel_graph = new_graph['bond_angle_graph']
    angle_dihedral_graph = new_graph["angle_dihedral_graph"]
    # new_positions, _ = move2origin(new_positions, batch, num_nodes)

    if update_target == "bond_length":
        new_target_values = get_bond_length(new_positions, feed_dict)

        atom_bond_graph.edge_feat['bond_length'] = new_target_values.squeeze()

    elif update_target == "bond_angle":
        new_target_values = get_bond_angle(new_positions, feed_dict)

        angle_atoms = atom_bond_graph.edges.gather(bond_angel_graph.edges.flatten()).reshape((-1, 4))
        mask = angle_atoms[:, 0] == angle_atoms[:, -1]
        bond_angel_graph.edge_feat['bond_angle'][~mask] = new_target_values.squeeze()

    elif update_target == "dihedral_angle":
        new_target_values = get_dihedral_angle(new_positions, feed_dict)

        # ultra_edges为所有的组成二面角的键角
        # super_edges为所有的组成键角的化学键
        ultra_edges = angle_dihedral_graph.edges
        super_edges = bond_angel_graph.edges
        edges = atom_bond_graph.edges

        # 过滤出首尾为同一化学键的可能
        head_edges = super_edges.gather(ultra_edges[:, 0])[:, 0]
        tail_edges = super_edges.gather(ultra_edges[:, 1])[:, -1]
        mask_1 = head_edges == tail_edges

        head_edge_atoms = edges.gather(head_edges)
        tail_edge_atoms = edges.gather(tail_edges)
        mask_2 = (head_edge_atoms[:, 0] == head_edge_atoms[:, 1]) | (tail_edge_atoms[:, 0] == tail_edge_atoms[:, 1])

        mask = mask_1 | mask_2
        angle_dihedral_graph.edge_feat["dihedral_angle"][~mask] = new_target_values.squeeze()

    return new_graph, new_positions, new_target_values


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
        prior_bond_length_list = []
        prior_bond_angle_list = []
        prior_dihedral_angle_list = []

        new_positions = prior_batch["positions"]

        for i, layer in enumerate(self.layers):
            delta_positions = layer(prior_graph)

            flag = i % 3

            prior_graph, new_positions, new_target_values = updated_graph(
                graph=prior_graph,
                feed_dict=prior_feed,
                now_positions=new_positions,
                delta_positions=delta_positions,
                batch=prior_batch["batch"],
                num_nodes=prior_batch["num_nodes"],
                update_target=TARGET_MAPPING[flag]
            )

            prior_positions_list.append(new_positions)

            if flag == 0:
                prior_bond_length_list.append(new_target_values)
            elif flag == 1:
                prior_bond_angle_list.append(new_target_values)
            else:
                prior_dihedral_angle_list.append(new_target_values)

        return prior_positions_list, prior_bond_length_list, prior_bond_angle_list, prior_dihedral_angle_list


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
                batch=encoder_batch["batch"],
                num_nodes=encoder_batch["num_nodes"],
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
                batch=decoder_batch["batch"],
                num_nodes=decoder_batch["num_nodes"],
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
        prior_positions_list, prior_bond_length_list, prior_bond_angle_list, prior_dihedral_angle_list = \
            self.prior(prior_graph, prior_feed, prior_batch)

        extra_output["prior_positions_list"] = prior_positions_list
        extra_output["prior_bond_length_list"] = prior_bond_length_list
        extra_output["prior_bond_angle_list"] = prior_bond_angle_list
        extra_output["prior_dihedral_angle_list"] = prior_dihedral_angle_list
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
            batch=prior_batch['batch'],
            num_nodes=prior_batch['num_nodes'],
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
        loss = 0
        loss_dict = {}

        # kld loss
        if "latent_mean" in extra_output:
            mean = extra_output["latent_mean"]
            log_std = extra_output["latent_logstd"]

            loss_kld, loss_dict_kld = self.compute_kld_loss(mean, log_std)
            loss += loss_kld
            loss_dict = {**loss_dict_kld}

        pos_x = encoder_batch["positions"]
        pos_y = encoder_batch["positions"]
        new_idx = self.update_iso(pos_x, pos_y, encoder_batch)

        # prior positions loss
        loss_tmp, _ = alignment_loss(
            # pos, pos_list[i].index_select(0, new_idx), batch, clamp=args.clamp_dist
            extra_output["gt_positions"],
            paddle.index_select(extra_output["prior_positions_list"][-1], axis=0, index=new_idx),
            extra_output["batch_dict"]
        )
        loss += loss_tmp * 4  # todo
        loss_dict["loss_prior_position"] = loss_tmp.numpy()[0]
        # prior positions loss

        # decoder positions loss
        for i, position in enumerate(decoder_positions_list):
            loss_tmp, _ = alignment_loss(
                extra_output["gt_positions"],
                paddle.index_select(position, axis=0, index=new_idx),
                extra_output["batch_dict"]
            )
            loss += loss_tmp * (1.0 if i == 0 else self.aux_weight)
            loss_dict[f"loss_decoder_position_{i}"] = loss_tmp.numpy()[0]

        loss_tmp, _ = alignment_loss(
            extra_output["gt_positions"],
            paddle.index_select(decoder_positions_list[-1], axis=0, index=new_idx),
            extra_output["batch_dict"]
        )
        loss += loss_tmp * 4 * self.n_layers
        loss_dict[f"loss_decoder_position"] = loss_tmp.numpy()[0]
        # decoder positions loss

        # geometry loss
        geometry_loss, loss_dict_geometry = self.compute_geometry_loss(extra_output)
        loss += geometry_loss

        loss_dict = {"loss": loss.numpy()[0], **loss_dict, **loss_dict_geometry}
        # loss_dict = {"loss": loss.numpy()[0], **loss_dict}

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

    def compute_kld_loss(self, mean, log_std):
        kld = -0.5 * paddle.sum(1 + 2 * log_std - mean.pow(2) - paddle.exp(2 * log_std), axis=-1)
        kld = kld.mean()  # todo check this line

        return kld * self.vae_beta, {'loss_kld': kld.numpy()[0]}

    def _compute_bond_length_loss(self, gt_bond_length, prior_bond_length_list, decoder_bond_length_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        bond_length_mask = ~gt_bond_length.equal(0)

        loss_prior = self.bond_length_loss(
            prior_bond_length_list[-1][bond_length_mask],
            gt_bond_length[bond_length_mask]

        )
        loss += (loss_prior * weight)  # todo
        loss_dict["loss_prior_bond_length"] = loss_prior.numpy()[0]

        for i, bond_length in enumerate(decoder_bond_length_list):
            loss_decoder = self.bond_length_loss(
                bond_length[bond_length_mask],
                gt_bond_length[bond_length_mask]
            )
            loss += (loss_decoder * weight * (1 + (i * step)))
            loss_dict[f"loss_decoder_bond_length_{i}"] = loss_decoder.numpy()[0]

        return loss, loss_dict

    def _compute_bond_angle_loss(self, gt_bond_angle, prior_bond_angle_list, decoder_bond_angle_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        bond_angle_mask = ~gt_bond_angle.equal(0)

        loss_prior = self.bond_angle_loss(
            prior_bond_angle_list[-1][bond_angle_mask],
            gt_bond_angle[bond_angle_mask]
        )
        loss += (loss_prior * weight)  # todo
        loss_dict["loss_prior_bond_angle"] = loss_prior.numpy()[0]

        for i, bond_angle in enumerate(decoder_bond_angle_list):
            loss_decoder = self.bond_length_loss(
                bond_angle[bond_angle_mask],
                gt_bond_angle[bond_angle_mask]
            )
            loss += (loss_decoder * weight * (1 + (i * step)))
            loss_dict[f"loss_decoder_bond_angle_{i}"] = loss_decoder.numpy()[0]

        return loss, loss_dict

    def _compute_dihedral_angle_loss(self, gt_dihedral_angle, prior_dihedral_angle_list, decoder_dihedral_angle_list, weight=1, step=0.2):
        loss = 0
        loss_dict = {}

        dihedral_angle_mask = ~gt_dihedral_angle.equal(0)

        loss_prior = self.dihedral_angle_loss(
            prior_dihedral_angle_list[-1].unsqueeze(1),
            gt_dihedral_angle
        )
        loss += (loss_prior * weight)  # todo
        loss_dict["loss_prior_dihedral_angle"] = loss_prior.numpy()[0]

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
            prior_bond_length_list=extra_output["prior_bond_length_list"],
            decoder_bond_length_list=extra_output["decoder_bond_length_list"],
            weight=3,
            step=0.2
        )

        loss_bond_angle, loss_dict_bond_angle = self._compute_bond_angle_loss(
            gt_bond_angle=extra_output["gt_bond_angle"],
            prior_bond_angle_list=extra_output["prior_bond_angle_list"],
            decoder_bond_angle_list=extra_output["decoder_bond_angle_list"],
            weight=2,
            step=0.2
        )

        loss_dihedral_angle, loss_dict_dihedral_angle = self._compute_dihedral_angle_loss(
            gt_dihedral_angle=extra_output["gt_dihedral_angle"],
            prior_dihedral_angle_list=extra_output["prior_dihedral_angle_list"],
            decoder_dihedral_angle_list=extra_output["decoder_dihedral_angle_list"],
            weight=1,
            step=0.2
        )

        loss = loss_bond_length + loss_bond_angle + loss_dihedral_angle
        # loss = loss_dihedral_angle
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
