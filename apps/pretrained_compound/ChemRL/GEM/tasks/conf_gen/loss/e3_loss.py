import numpy as np
import paddle


def move2origin(poses, batch):
    new_pos = []
    pos_mean = []

    for i in range(batch.max() + 1):
        pos = poses[batch == i]
        _mean = pos.mean(axis=0)

        new_pos.append(pos - _mean)
        pos_mean.append(_mean)

    return paddle.concat(new_pos), paddle.stack(pos_mean)


def compute_loss(extra_output, pos_list, gt_pos, batch, args):
    loss_dict = {}
    loss = 0

    loss_tmp, _ = alignment_loss(
        gt_pos, extra_output["prior_pos_list"][-1], batch
    )
    loss = loss + loss_tmp
    loss_dict["loss_prior_pos"] = loss_tmp.numpy()[0]

    mean = extra_output["latent_mean"]
    log_std = extra_output["latent_logstd"]
    kld = -0.5 * paddle.sum(1 + 2 * log_std - mean.pow(2) - paddle.exp(2 * log_std), axis=-1)
    kld = kld.mean()
    loss = loss + kld * args.vae_beta
    loss_dict["loss_kld"] = kld.numpy()[0]

    loss_tmp, _ = alignment_loss(
        gt_pos, pos_list[-1], batch
    )
    loss = loss + loss_tmp
    loss_dict["loss_pos_last"] = loss_tmp.numpy()[0]

    if args.aux_loss > 0:
        for i in range(len(pos_list) - 1):
            loss_tmp, _ = alignment_loss(gt_pos, pos_list[i], batch)
            loss = loss + loss_tmp * (args.aux_loss if i < len(pos_list) else 1.0)
            loss_dict[f"loss_pos_{i}"] = loss_tmp.numpy()[0]

    # if args.ang_lam > 0 or args.bond_lam > 0:
    #     bond_loss, angle_loss = self.aux_loss(pos, pos_list[-1].index_select(0, new_idx), batch)
    #     loss_dict["bond_loss"] = bond_loss
    #     loss_dict["angle_loss"] = angle_loss
    #     loss = loss + args.bond_lam * bond_loss + args.ang_lam * angle_loss

    loss_dict["loss"] = loss.numpy()[0]
    return loss, loss_dict


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


def alignment_loss(pos_y, pos_x, batch, clamp=None):
    with paddle.no_grad():
        num_nodes = batch["num_nodes"]
        total_nodes = pos_y.shape[0]
        num_graphs = len(batch["mols"])

        y, pos_y_mean = move2origin(pos_y, batch["batch"])
        x, pos_x_mean = move2origin(pos_x, batch["batch"])

        a = y + x
        b = y - x
        a = a.reshape((total_nodes, 1, 3))
        b = b.reshape((total_nodes, 3, 1))

        tmp0 = paddle.concat(
            [
                paddle.expand(paddle.zeros((1, 1, 1)), (total_nodes, -1, -1)),
                -b.transpose((0, 2, 1))
            ], axis=-1
        )
        eye = paddle.expand(paddle.eye(3).unsqueeze(0), (total_nodes, -1, -1))
        a = paddle.expand(a, (-1, 3, -1))
        tmp1 = paddle.cross(eye, a, axis=-1)
        tmp1 = paddle.concat([b, tmp1], axis=-1)
        tmp = paddle.concat([tmp0, tmp1], axis=1)
        tmpb = paddle.bmm(tmp.transpose((0, 2, 1)), tmp).reshape((total_nodes, -1))
        tmpb = move2origin(tmpb, batch["batch"])[1].reshape((num_graphs, 4, 4))
        w, v = paddle.linalg.eigh(tmpb)
        min_rmsd = w[:, 0]
        min_q = v[:, :, 0]
        rotation = quaternion_to_rotation_matrix(min_q)
        t = pos_y_mean - paddle.einsum("kj,kij->ki", pos_x_mean, rotation)
        rotation = paddle.to_tensor(np.repeat(rotation.numpy(), batch["num_nodes"], axis=0))
        t = paddle.to_tensor(np.repeat(t.numpy(), batch["num_nodes"], axis=0))

    pos_x = paddle.einsum("kj,kij->ki", pos_x, rotation) + t
    if clamp is None:
        loss = move2origin((pos_y - pos_x).norm(axis=-1, keepdim=True), batch["batch"])[1].mean()
    else:
        loss = move2origin((pos_y - pos_x).norm(axis=-1, keepdim=True), batch["batch"])[1]
        loss = paddle.clip(loss, min=clamp).mean()
    return loss, min_rmsd.mean()
