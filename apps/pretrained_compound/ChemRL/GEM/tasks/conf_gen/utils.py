import copy
import math

import numpy as np
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import paddle


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                    self.eta_min
                    + (1 - self.eta_min)
                    * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                    / 2
            )

        else:
            return self.eta_min + self.lr_step * step

def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def set_rdmol_positions(rdkit_mol, pos):
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = copy.deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_best_rmsd(gen_mol, ref_mol, use_ff=True):
    if use_ff:
        MMFFOptimizeMolecule(gen_mol)
    gen_mol = RemoveHs(gen_mol)
    ref_mol = RemoveHs(ref_mol)
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def get_rmsd_min(inputargs):
    mols, use_ff, threshold = inputargs
    gen_mols, ref_mols = mols
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def scatter_sum(src, index, dim, dim_size):
    index = index.unsqueeze(-1)
    index = paddle.expand(index, shape=src.shape)
    i, j = index.shape

    size = list(src.shape)

    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = paddle.zeros(size, dtype=src.dtype)

    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))

    if dim == 0:
        index = paddle.stack([index.flatten(), grid_y.flatten()], axis=1)
    else:
        index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)

    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(src, index=updates_index)
    return paddle.scatter_nd_add(out, index, updates)


def scatter_mean(src, index, dim, dim_size):
    out = scatter_sum(src, index, dim, dim_size)
    dim_size = out.shape[0]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones.unsqueeze(-1), index, index_dim, dim_size)

    count[count < 1] = 1
    count = paddle.expand(count, shape=out.shape)

    return paddle.divide(out, count)


if __name__ == "__main__":
    import numpy as np

    _src = paddle.to_tensor(np.load('./data/example/src.npy'))
    _index = paddle.to_tensor(np.load('./data/example/index.npy'))

    scatter_mean(_src, _index, 0, 128)

    print("DONE!!!")
