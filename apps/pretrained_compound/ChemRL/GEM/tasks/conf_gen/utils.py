import copy

import numpy as np
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

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


def get_best_rmsd(gen_mol, ref_mol):
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