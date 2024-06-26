import argparse
import copy
import multiprocessing
import os
import pickle
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LambdaDecay

import sys

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import set_rdmol_positions, get_best_rmsd, get_rmsd_min

sys.path.append("..")

from conf_gen.model.vae import VAE
from conf_gen.datasets import load_mol_to_dataset, ConfGenTaskTransformFn, ConfGenTaskCollateFn
from pahelix.utils import load_json_config
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel

from tqdm import tqdm

from rdkit import Chem


def get_steps_per_epoch(train_num, args):
    """tbd"""
    # add as argument
    if args.debug:
        train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    return steps_per_epoch


@paddle.no_grad()
def evaluate(model: VAE, data_gen, args):
    model.eval()

    pbar = tqdm(data_gen, desc="Training Iteration", disable=False)

    inchikey2pairs = dict()

    for step, (
            prior_graph, encoder_graph, decoder_graph,
            prior_feed, encoder_feed, decoder_feed,
            prior_batch, encoder_batch, decoder_batch,
            mol_list
    ) in enumerate(pbar, start=1):

        # graph to tensor
        for k in prior_graph:
            prior_graph[k] = prior_graph[k].tensor()
        for k in encoder_graph:
            encoder_graph[k] = encoder_graph[k].tensor()
        for k in decoder_graph:
            decoder_graph[k] = decoder_graph[k].tensor()
        # graph to tensor

        ###############################

        # feed to tensor
        for k in prior_feed:
            prior_feed[k] = paddle.to_tensor(prior_feed[k])
        for k in encoder_feed:
            encoder_feed[k] = paddle.to_tensor(encoder_feed[k])
        for k in decoder_feed:
            decoder_feed[k] = paddle.to_tensor(decoder_feed[k])
        # feed to tensor

        ###############################

        # batch to tensor
        for k in prior_batch:
            prior_batch[k] = paddle.to_tensor(prior_batch[k])
        for k in encoder_batch:
            encoder_batch[k] = paddle.to_tensor(encoder_batch[k])
        for k in decoder_batch:
            decoder_batch[k] = paddle.to_tensor(decoder_batch[k])
        # batch to tensor

        net_inputs = {
            "prior_graph": prior_graph,
            "prior_feed": prior_feed,
            "prior_batch": prior_batch,
            "decoder_graph": decoder_graph,
            "decoder_feed": decoder_feed,
            "decoder_batch": decoder_batch,
            "encoder_graph": encoder_graph,
            "encoder_feed": encoder_feed,
            "encoder_batch": encoder_batch,
        }

        with paddle.no_grad():
            prior_positions_list, positions_list = model(**net_inputs, sample=True, compute_loss=False)

            num_nodes = prior_batch["num_nodes"]
            pre_nodes = 0

            for i, mol in enumerate(mol_list):
                inchikey = Chem.MolToInchiKey(mol)

                if inchikey not in inchikey2pairs:
                    n = len(prior_positions_list + positions_list) + 1
                    inchikey2pairs[inchikey] = [[] for _ in range(n)]

                inchikey2pairs[inchikey][-1].append(mol)

                for j, pos in enumerate(prior_positions_list + positions_list):
                    tmp_mol = set_rdmol_positions(mol, pos[pre_nodes: pre_nodes + num_nodes[i]])
                    inchikey2pairs[inchikey][j].append(tmp_mol)

                pre_nodes += num_nodes[i]

        with paddle.no_grad():
            prior_positions_list, positions_list = model(**net_inputs, sample=True, compute_loss=False)

            num_nodes = prior_batch["num_nodes"]
            pre_nodes = 0

            for i, mol in enumerate(mol_list):
                inchikey = Chem.MolToInchiKey(mol)

                for j, pos in enumerate(prior_positions_list + positions_list):
                    tmp_mol = set_rdmol_positions(mol, pos[pre_nodes: pre_nodes + num_nodes[i]])
                    inchikey2pairs[inchikey][j].append(tmp_mol)

                pre_nodes += num_nodes[i]

    if not args.debug:
        del_inchikey = []
        for inchikey in inchikey2pairs.keys():
            if len(inchikey2pairs[inchikey][-1]) < 50 or len(inchikey2pairs[inchikey][-1]) > 500:
                del_inchikey.append(inchikey)
        for inchikey in del_inchikey:
            del inchikey2pairs[inchikey]

    cov_list = []
    mat_list = []

    def input_args():
        for _inchikey in inchikey2pairs.keys():
            yield inchikey2pairs[_inchikey], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25

    history = [[] for _ in range(2 * args.num_layers + 2)]

    for mols, use_ff, threshold in input_args():
        cov, mat, min_index = get_rmsd_min(mols[-2:], use_ff, threshold)

        cov_list.append(cov)
        mat_list.append(mat)

        for i, _mol_list in enumerate(mols[:-1]):
            history[i] += np.array(mols[i])[list(min_index)].tolist()
        else:
            history[-1] += mols[-1]

    print(f"cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
    print(f"mat mean {np.mean(mat_list)} med {np.median(mat_list)}")

    return np.mean(cov_list), np.mean(mat_list), history


def main(args):
    prior_config = load_json_config(args.prior_config)
    encoder_config = load_json_config(args.encoder_config)
    decoder_config = load_json_config(args.decoder_config)
    head_config = load_json_config(args.head_config)

    done_file = os.path.join(args.cached_data_path, f"{args.dataset_name}_eval.done")

    if args.cached_data_path == "" or (args.cached_data_path != "" and not os.path.exists(done_file)):
        print("===> Converting data...")
        test_dataset = load_mol_to_dataset(args.test_data_path, debug=args.debug)

        if args.debug:
            args.epochs = 10
            args.dataset = 'debug'
            test_dataset = test_dataset[:32]
            args.num_workers = 1

        print({"test_num": len(test_dataset)})

        transform_fn = ConfGenTaskTransformFn(n_noise_mol=args.n_noise_mol, isomorphism=False)
        test_dataset.transform(transform_fn, num_workers=args.num_workers)

        if args.cached_data_path and not args.debug:

            print("===> Save data ...")

            if not os.path.exists(args.cached_data_path):
                os.makedirs(args.cached_data_path)

            with open(os.path.join(args.cached_data_path, 'test.npy'), 'wb') as w1:
                pickle.dump(test_dataset, w1)

            with open(done_file, 'w') as w2:
                w2.write("DONE!")

    else:
        print('====> Read preprocessing data...')

        with open(os.path.join(args.cached_data_path, 'test.npy'), 'rb') as f1:
            test_dataset = pickle.load(f1)

    collate_fn = ConfGenTaskCollateFn(
        atom_names=encoder_config['atom_names'],
        bond_names=encoder_config['bond_names'],
        bond_float_names=encoder_config['bond_float_names'],
        bond_angle_float_names=encoder_config['bond_angle_float_names'],
        dihedral_angle_float_names=encoder_config['dihedral_angle_float_names'],
        isomorphism=False
    )

    test_data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = VAE(
        prior_config=prior_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        head_config=head_config,
        n_layers=args.num_layers
    )

    if not args.init_model is None and not args.init_model == "":
        # compound_encoder.set_state_dict(paddle.load(args.init_model))
        model.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    print("Evaluating...")
    _, _, history = evaluate(model, test_data_gen, args)

    if args.save_conf_dir:
        for i, history_mols in enumerate(history):
            if i != len(history) - 1:
                writer = Chem.SDWriter(f'{args.save_conf_dir}/{args.dataset_name}_step_{i}.sdf')
            else:
                writer = Chem.SDWriter(f'{args.save_conf_dir}/{args.dataset_name}_ref.sdf')

            for mol in history_mols:
                writer.write(mol)

            writer.close()


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--dataset_name", type=str, default='qm9')
    parser.add_argument("--test_data_path", type=str)

    parser.add_argument("--prior_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--head_config", type=str)

    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--n_noise_mol", type=int, default=1)

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument("--cached_data_path", type=str, default='')
    parser.add_argument("--save_conf_dir", type=str, default='')

    args = parser.parse_args()
    args.use_ff = True
    print(args)

    main(args)


if __name__ == "__main__":
    main_cli()
