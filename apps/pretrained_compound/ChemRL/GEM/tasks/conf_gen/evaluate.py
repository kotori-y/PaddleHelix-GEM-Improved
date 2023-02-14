import argparse
import pickle
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist

from datasets import load_sdf_mol_to_dataset, load_pickled_mol_to_dataset, ConfGenTaskTransformFn, ConfGenTaskCollateFn
from loss.e3_loss import compute_loss
from model.gnn import ConfGenModel
from pahelix.utils import load_json_config
from utils import exempt_parameters, set_rdmol_positions, get_best_rmsd

from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel
import multiprocessing
from rdkit import Chem


def get_steps_per_epoch(train_num, args):
    """tbd"""
    # add as argument
    if args.debug:
        train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    return steps_per_epoch


def evaluate(args, model, test_data_gen, test_num, model_dir=None):
    model.eval()

    steps = get_steps_per_epoch(test_num, args)
    step = 0

    mol_labels = []
    mol_preds = []

    for net_input, batch, prior_pos_list, gt_pos_list in test_data_gen:
        net_input = {k: v.tensor() if v is not None else v for k, v in net_input.items()}

        gt_pos_list = paddle.to_tensor(np.vstack(gt_pos_list), dtype=paddle.float32)
        prior_pos_list = paddle.to_tensor(np.vstack(prior_pos_list), dtype=paddle.float32)

        net_input["batch"] = batch
        net_input["prior_poses"] = prior_pos_list

        batch_size = len(batch["mols"])
        n_nodes = batch["num_nodes"].tolist()

        for _ in range(2):

            with paddle.no_grad():
                _, pos_list = model(**net_input, sample=True)
            pred = pos_list[-1]

            pre_nodes = 0
            for i in range(batch_size):
                mol_labels.append(batch["mols"][i])
                mol_preds.append(
                    set_rdmol_positions(
                        batch["mols"][i], pred[pre_nodes: pre_nodes + n_nodes[i]]
                    )
                )
                pre_nodes += n_nodes[i]

        step += 1
        if step > steps:
            print("jumpping out")
            break

    smiles2pairs = dict()
    for gen_mol in mol_preds:
        smiles = Chem.MolToSmiles(gen_mol)
        if smiles not in smiles2pairs:
            smiles2pairs[smiles] = [[gen_mol]]
        else:
            smiles2pairs[smiles][0].append(gen_mol)
    for ref_mol in mol_labels:
        smiles = Chem.MolToSmiles(ref_mol)
        if len(smiles2pairs[smiles]) == 1:
            smiles2pairs[smiles].append([ref_mol])
        else:
            smiles2pairs[smiles][1].append(ref_mol)

    del_smiles = []
    for smiles in smiles2pairs.keys():
        if len(smiles2pairs[smiles][1]) < 50 or len(smiles2pairs[smiles][1]) > 500:
            del_smiles.append(smiles)
    for smiles in del_smiles:
        del smiles2pairs[smiles]

    cov_list = []
    mat_list = []
    pool = multiprocessing.Pool(args.workers)

    def input_args():
        for smiles in smiles2pairs.keys():
            yield smiles2pairs[smiles], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25

    for res in pool.imap(get_best_rmsd, input_args(), chunksize=10):
        cov_list.append(res[0])
        mat_list.append(res[1])

    print(f"cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
    print(f"mat mean {np.mean(mat_list)} med {np.median(mat_list)}")
    return np.mean(cov_list), np.mean(mat_list)


def main(args):
    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)

    if args.dropout_rate is not None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate
        model_config['dropout_rate'] = args.dropout_rate

    prior_model = GNNModel(compound_encoder_config)
    encoder_model = GNNModel(compound_encoder_config)
    decoder_model = GeoGNNModel(compound_encoder_config)
    model = ConfGenModel(model_config, compound_encoder_config, prior_model, encoder_model, decoder_model)

    print('Total param num: %s' % (len(model.parameters())))

    # if args.distributed:
    #     model = paddle.DataParallel(model)

    test_dataset = load_pickled_mol_to_dataset(args.test_dataset)
    test_num = len(test_dataset)

    if args.debug:
        test_dataset = test_dataset[:64]

    transform_fn = ConfGenTaskTransformFn(is_inference=True)
    test_dataset.transform(transform_fn, num_workers=args.num_workers)

    collate_fn = ConfGenTaskCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'])

    test_data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    print("Evaluating...")
    evaluate(args, model, test_data_gen, test_num)


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--test_dataset", type=str, default=None)

    parser.add_argument("--cached_data_path", type=str, default=None)

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--vae_beta", type=float, default=1)
    parser.add_argument("--aux_loss", type=float, default=0.2)
    parser.add_argument("--num_message_passing_steps", type=int, default=3)

    parser.add_argument("--dataset_name", type=str, default="qm9")

    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    main_cli()
