import copy

import numpy as np
import paddle
import paddle.nn.functional as F
try:
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean, set_rdmol_positions
except:
    from conf_gen.utils import scatter_mean, set_rdmol_positions
import multiprocessing as mp
from rdkit.Chem.AllChem import AlignMol
import paddle.nn as nn


class ParallelAlignMol(object):

    def __init__(self, num_workers=32):
        super().__init__()
        self.pool = mp.Pool(num_workers)

    @staticmethod
    def _align_mol(mol_truth, mol_gen):
        mol_truth = copy.deepcopy(mol_truth)
        # mol_truth = RemoveHs(copy.deepcopy(mol_truth))
        # mol_gen = RemoveHs(copy.deepcopy(mol_gen))
        AlignMol(mol_truth, mol_gen)
        return mol_truth.GetConformer(0).GetPositions() # (num_atoms, 3)

    def __call__(self, mols_truth, mols_gen):
        probes = self.pool.starmap(ParallelAlignMol._align_mol, zip(mols_truth, mols_gen))
        return paddle.to_tensor(np.vstack(probes), dtype=paddle.float32)


def move2origin(poses, batch, num_nodes):
    dim_size = batch.max() + 1
    index = paddle.to_tensor(batch)
    poses_mean = scatter_mean(poses, index, 0, dim_size)
    _poses_mean = poses_mean.numpy().repeat(num_nodes, axis=0)
    _poses_mean = paddle.to_tensor(_poses_mean, dtype=poses_mean.dtype)
    return poses - _poses_mean, poses_mean


def rmsd_loss(pos, batch):
    mols_truth = []
    mols_gen = []
    for i, mol in enumerate(batch["mols"]):
        m = copy.deepcopy(mol)
        mols_truth.append(m)
        conf = pos[batch["batch"] == i]
        mols_gen.append(set_rdmol_positions(m, conf.clone().detach()))
    probe = ParallelAlignMol(num_workers=32)(mols_truth, mols_gen)  # (\sum_G num_atoms_of_G, 3)
    loss_tmp = scatter_mean(((probe - pos) ** 2).sum(-1).unsqueeze(-1), paddle.to_tensor(batch["batch"]), dim=0, dim_size=len(batch["mols"])) ** (0.5)
    return loss_tmp.sum() / len(batch["mols"])


def compute_loss(extra_output, feed_dict, gt_pos, pos_list, batch, args):
    # bar_loss = nn.SmoothL1Loss()
    # blr_loss = nn.SmoothL1Loss()

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
    kld = kld.sum() / len(batch["mols"])
    loss = loss + kld * args.vae_beta
    loss_dict["loss_kld"] = kld.numpy()[0]

    loss_tmp, _ = alignment_loss(
        gt_pos, pos_list[-1], batch
    )
    loss = loss + loss_tmp
    loss_dict["loss_pos_last"] = loss_tmp.numpy()[0]

    if args.aux_loss > 0:
        for i in range(len(pos_list) - 1):
            _loss_tmp, _ = alignment_loss(gt_pos, pos_list[i], batch)
            loss = loss + _loss_tmp * (args.aux_loss if i < len(pos_list) else 1.0)
            loss_dict[f"loss_pos_{i}"] = _loss_tmp.numpy()[0]

    # loss_bar = bar_loss(extra_output['bar_list'], feed_dict['Ba_bond_angle'] / np.pi)
    # loss_blr = blr_loss(extra_output['blr_list'], feed_dict['Bl_bond_length'])
    # loss += (loss_bar + loss_blr)
    # loss_dict["loss_bar_last"] = loss_bar.numpy()[0]
    # loss_dict["loss_blr_last"] = loss_blr.numpy()[0]

    # assert len(aux_dict['bar_list']) == len(aux_dict['blr_list'])
    # for i in range(len(aux_dict['bar_list']) - 1):
    #     _loss_bar = bar_loss(aux_dict['bar_list'][i], feed_dict['Ba_bond_angle'] / np.pi)
    #     _loss_blr = blr_loss(aux_dict['blr_list'][i], feed_dict['Bl_bond_length'])
    #
    #     loss = loss + _loss_bar * (args.aux_loss if i < len(pos_list) else 1.0)
    #     loss = loss + _loss_blr * (args.aux_loss if i < len(pos_list) else 1.0)
    #
    #     loss_dict[f"loss_bar_{i}"] = _loss_bar.numpy()[0]
    #     loss_dict[f"loss_blr_{i}"] = _loss_blr.numpy()[0]

    # if args.ang_lam > 0 or args.bond_lam > 0:
    #     bond_loss, angle_loss = aux_loss(gt_pos, pos_list[-1], batch)
    #     loss_dict["bond_loss"] = bond_loss
    #     loss_dict["angle_loss"] = angle_loss
    #     loss = loss + args.bond_lam * bond_loss + args.ang_lam * angle_loss

    loss_dict["loss"] = loss.numpy()[0]
    return loss, loss_dict


def compute_vae_kl(mu_q, logvar_q, mu_prior, logvar_prior):
    mu1 = mu_q
    std1 = paddle.exp(0.5*logvar_q)
    mu2 = mu_prior
    std2 = paddle.exp(0.5*logvar_prior)
    kl = - 0.5 + paddle.log(std2 / (std1 + 1e-8) + 1e-8) + \
        ((paddle.pow(std1, 2) + paddle.pow(mu1 - mu2, 2)) / (2 * paddle.pow(std2, 2)))

    return kl


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
        num_graphs = num_nodes.shape[0]

        y, pos_y_mean = move2origin(pos_y, batch["batch"], num_nodes)
        x, pos_x_mean = move2origin(pos_x, batch["batch"], num_nodes)

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
        tmpb = move2origin(tmpb, batch["batch"], num_nodes)[1].reshape((num_graphs, 4, 4))
        w, v = paddle.linalg.eigh(tmpb)
        min_rmsd = w[:, 0]
        min_q = v[:, :, 0]
        rotation = quaternion_to_rotation_matrix(min_q)
        t = pos_y_mean - paddle.einsum("kj,kij->ki", pos_x_mean, rotation)
        rotation = paddle.to_tensor(np.repeat(rotation.numpy(), batch["num_nodes"], axis=0))
        t = paddle.to_tensor(np.repeat(t.numpy(), batch["num_nodes"], axis=0))

    pos_x = paddle.einsum("kj,kij->ki", pos_x, rotation) + t
    if clamp is None:
        loss = move2origin((pos_y - pos_x).norm(axis=-1, keepdim=True), batch["batch"], num_nodes)[1].mean()
    else:
        loss = move2origin((pos_y - pos_x).norm(axis=-1, keepdim=True), batch["batch"], num_nodes)[1]
        loss = paddle.clip(loss, min=clamp).mean()
    return loss, min_rmsd.mean()


def aux_loss(pos_y, pos_x, batch):
    edge_index = batch.edge_index
    src = edge_index[0]
    tgt = edge_index[1]
    true_bond = paddle.norm(pos_y[src] - pos_y[tgt], axis=-1)
    pred_bond = paddle.norm(pos_x[src] - pos_x[tgt], axis=-1)
    bond_loss = paddle.mean(F.l1_loss(pred_bond, true_bond))

    nei_src_index = batch.nei_src_index.view(-1)
    nei_tgt_index = batch.nei_tgt_index
    nei_tgt_mask = batch.nei_tgt_mask

    random_tgt_index = pos_y.new_zeros(nei_tgt_index.size()).uniform_()
    random_tgt_index = paddle.where(
        nei_tgt_mask, pos_y.new_zeros(nei_tgt_index.size()), random_tgt_index
    )
    random_tgt_index_sort = paddle.sort(random_tgt_index, descending=True, axis=0)[1][:2]
    tgt_1, tgt_2 = random_tgt_index_sort[0].unsqueeze(0), random_tgt_index_sort[1].unsqueeze(0)

    tgt_1 = paddle.gather(nei_tgt_index, 0, tgt_1).view(-1)
    tgt_2 = paddle.gather(nei_tgt_index, 0, tgt_2).view(-1)

    def get_angle(vec1, vec2):
        vec1 = vec1 / (paddle.norm(vec1, keepdim=True, axis=-1) + 1e-6)
        vec2 = vec2 / (paddle.norm(vec2, keepdim=True, axis=-1) + 1e-6)
        return paddle.einsum("nc,nc->n", vec1, vec2)

    true_angle = get_angle(
        pos_y[tgt_1] - pos_y[nei_src_index], pos_y[tgt_2] - pos_y[nei_src_index]
    )
    pred_angle = get_angle(
        pos_x[tgt_1] - pos_x[nei_src_index], pos_x[tgt_2] - pos_x[nei_src_index]
    )
    angle_loss = paddle.mean(F.l1_loss(pred_angle, true_angle))

    return bond_loss, angle_loss
