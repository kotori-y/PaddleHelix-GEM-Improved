import argparse
import copy
import os
import pickle
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LambdaDecay

import sys

from apps.pretrained_compound.ChemRL.GEM.src.utils import exempt_parameters
from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import set_rdmol_positions, get_best_rmsd
from pahelix.datasets import InMemoryDataset

sys.path.append("..")

from conf_gen.model.vae import VAE
from conf_gen.datasets import load_mol_to_dataset, ConfGenTaskTransformFn, ConfGenTaskCollateFn
from pahelix.utils import load_json_config
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel

from tqdm import tqdm

# from loss.e3_loss import compute_loss
# # from model.gnn import ConfGenModel
# from pahelix.utils import load_json_config
# from utils import exempt_parameters, set_rdmol_positions, get_best_rmsd, WarmCosine
#
# from pahelix.model_zoo.gem_model import GeoGNNModel, GeoPredModel


def train(model: VAE, opt, data_gen, args):
    model.train()
    print(f"lr: {opt.get_lr()}")

    loss_accum_dict = defaultdict(float)

    pbar = tqdm(data_gen, desc="Training Iteration", disable=args.disable_tqdm)

    for step, (
            prior_graph, encoder_graph, decoder_graph,
            prior_feed, encoder_feed, decoder_feed,
            prior_batch, encoder_batch, decoder_batch,
            _
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

        loss, loss_dict, _ = model(**net_inputs)

        loss.backward()
        opt.step()
        opt.clear_grad()
        # scheduler.step()

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step
    return loss_accum_dict


@paddle.no_grad()
def evaluate(model: VAE, data_gen, args):
    model.eval()

    loss_accum_dict = defaultdict(float)

    mol_labels = []
    mol_preds = []

    pbar = tqdm(data_gen, desc="Evaluating Iteration", disable=args.disable_tqdm)

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
            _, loss_dict, positions_list = model(**net_inputs)

        position = positions_list[-1]
        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

        # batch_size = len(prior_batch["num_nodes"])
        n_nodes = prior_batch["num_nodes"].tolist()
        pre_nodes = 0

        for i, mol in enumerate(mol_list):
            mol_labels.append(mol)

            mol_pred = set_rdmol_positions(mol, position[pre_nodes: pre_nodes + n_nodes[i]])
            mol_preds.append(mol_pred)

            pre_nodes += n_nodes[i]

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step

    rmsd_list = []
    for gen_mol, ref_mol in zip(mol_preds, mol_labels):
        try:
            rmsd_list.append(get_best_rmsd(gen_mol, ref_mol))
        except Exception as e:
            continue

    return loss_accum_dict, np.mean(rmsd_list)


def main(args):
    prior_config = load_json_config(args.prior_config)
    encoder_config = load_json_config(args.encoder_config)
    decoder_config = load_json_config(args.decoder_config)
    # aux_config = load_json_config(args.aux_config)
    head_config = load_json_config(args.head_config)

    if args.dropout_rate is not None:
        prior_config['dropout_rate'] = args.dropout_rate
        encoder_config['dropout_rate'] = args.dropout_rate
        decoder_config['dropout_rate'] = args.dropout_rate
        # down_config['dropout_rate'] = args.dropout_rate

    done_file = os.path.join(args.cached_data_path, f"{args.dataset}.done")

    if args.distributed or args.cached_data_path == "" or (args.cached_data_path != "" and not os.path.exists(done_file)):

        print("===> Convert data...")

        train_dataset = load_mol_to_dataset(args.train_data_path, debug=args.debug)
        valid_dataset = load_mol_to_dataset(args.valid_data_path, debug=args.debug)

        if args.debug:
            train_dataset = train_dataset[:32]
            args.epochs = 10
            args.dataset = 'debug'
            valid_dataset = valid_dataset[:32]
            args.num_workers = 1
            args.cached_data_path = "./data/cached_data/debug"

        print({
            "train_num": len(train_dataset),
            "valid_num": len(valid_dataset),
        })

        train_dataset = train_dataset[dist.get_rank()::dist.get_world_size()]
        valid_dataset = valid_dataset[dist.get_rank()::dist.get_world_size()]

        transform_fn = ConfGenTaskTransformFn(n_noise_mol=args.n_noise_mol, isomorphism=args.isomorphism)

        train_dataset.transform(transform_fn, num_workers=args.num_workers)
        valid_dataset.transform(transform_fn, num_workers=args.num_workers)

        if args.cached_data_path and not args.debug:

            print("===> Save data ...")

            if not os.path.exists(args.cached_data_path):
                os.makedirs(args.cached_data_path)

            with open(os.path.join(args.cached_data_path, 'train.npy'), 'wb') as w1:
                pickle.dump(train_dataset, w1)

            with open(os.path.join(args.cached_data_path, 'valid.npy'), 'wb') as w2:
                pickle.dump(valid_dataset, w2)

            with open(done_file, 'w') as w3:
                w3.write("DONE!")

    else:
        print('====> Read preprocessing data...')

        with open(os.path.join(args.cached_data_path, 'train.npy'), 'rb') as f1:
            train_dataset = pickle.load(f1)

        with open(os.path.join(args.cached_data_path, 'valid.npy'), 'rb') as f2:
            valid_dataset = pickle.load(f2)

    collate_fn = ConfGenTaskCollateFn(
        atom_names=encoder_config['atom_names'],
        bond_names=encoder_config['bond_names'],
        bond_float_names=encoder_config['bond_float_names'],
        bond_angle_float_names=encoder_config['bond_angle_float_names'],
        dihedral_angle_float_names=encoder_config['dihedral_angle_float_names'],
        isomorphism=args.isomorphism
        # pretrain_tasks=head_config['pretrain_tasks']
    )

    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_data_gen = valid_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )

    compound_encoder = GeoGNNModel(prior_config)
    if not args.init_model is None and not args.init_model == "":
        compound_encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)

    model = VAE(
        prior_config=prior_config,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        head_config=head_config,
        n_layers=args.num_layers,
        vae_beta=args.vae_beta,
        isomorphism=args.isomorphism,
        compound_encoder=compound_encoder
    )

    model_without_ddp = model
    args.disable_tqdm = False

    if args.distributed:
        model = paddle.DataParallel(model, find_unused_parameters=True)

    model_params = model_without_ddp.parameters()
    head_params = exempt_parameters(model_params, compound_encoder.parameters())

    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=head_params,
        weight_decay=args.weight_decay
    )
    num_params = sum(p.numel() for p in head_params)
    print(f"#Params: {num_params}")

    train_history = []
    valid_history = []

    root = os.path.join(args.log_path, args.task)
    if not os.path.exists(root):
        os.makedirs(root)

    for epoch in range(args.epochs):
        if not args.distributed or dist.get_rank() == 0:
            print("\n=====Epoch {:0>3}".format(epoch))
            print(f"Training at 0 / {dist.get_world_size()}...")

        loss_dict_train = train(model, optimizer, train_data_gen, args)
        train_history.append(loss_dict_train)

        with open(os.path.join(root, f"./train_history_{dist.get_rank()}.pkl"), 'wb') as f:
            pickle.dump(train_history, f)

        if not args.distributed or dist.get_rank() == 0:
            print("Validating...")
            loss_dict_valid, valid_rmsd = evaluate(model=model, data_gen=valid_data_gen, args=args)
            loss_dict_valid['rmsd'] = valid_rmsd

            valid_history.append(loss_dict_valid)
            with open(os.path.join(root, "./valid_history.pkl"), 'wb') as f:
                pickle.dump(valid_history, f)

            print(f"====== epoch: {epoch:0>3} ======: \n train: {loss_dict_train['loss']} \n valid: {loss_dict_valid['loss']}")
            paddle.save(model.state_dict(), os.path.join(root, 'params', f'iter_{epoch}', 'model.pdparams'))


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--dataset", type=str, default='debug')
    parser.add_argument("--task", type=str, default='debug')

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--log_path", type=str, default='./log')

    parser.add_argument("--prior_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--head_config", type=str)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--n_noise_mol", type=int, default=1)
    parser.add_argument("--vae_beta", type=float, default=1.0)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--isomorphism", action="store_true", default=False)

    parser.add_argument("--init_model", type=str, default='')

    parser.add_argument("--cached_data_path", type=str, default='')

    args = parser.parse_args()
    print(args)

    if args.distributed:
        dist.init_parallel_env()

    main(args)


if __name__ == "__main__":
    main_cli()
