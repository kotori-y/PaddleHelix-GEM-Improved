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
            prior_batch, encoder_batch, decoder_batch
    ) in enumerate(pbar):

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

        loss, loss_dict, _ = model(
            prior_graph, encoder_graph, decoder_graph,
            prior_feed, encoder_feed, decoder_feed,
            prior_batch, encoder_batch, decoder_batch
        )

        loss.backward()
        opt.step()
        opt.clear_grad()
        # scheduler.step()

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def main(args):
    prior_config = load_json_config(args.prior_config)
    encoder_config = load_json_config(args.encoder_config)
    decoder_config = load_json_config(args.decoder_config)
    # aux_config = load_json_config(args.aux_config)
    head_config = load_json_config(args.head_config)

    # if args.dropout_rate is not None:
    #     encoder_config['dropout_rate'] = args.dropout_rate
    #     decoder_config['dropout_rate'] = args.dropout_rate
    #     # down_config['dropout_rate'] = args.dropout_rate

    if not args.distributed or dist.get_rank() == 0:
        print("===> Converting data...")

        train_dataset = load_mol_to_dataset(args.data_path, debug=args.debug)
        # valid_dataset = load_mol_to_dataset(args.valid_dataset)
        # test_dataset = load_mol_to_dataset(args.test_dataset)

        print({
            "train_num": len(train_dataset),
            # "valid_num": len(valid_dataset),
            # "test_num": len(test_dataset),
        })

        if args.debug:
            train_dataset = train_dataset[:32]
            args.epochs = 10
            args.dataset = 'debug'
            # valid_dataset = valid_dataset[:32]
            # test_dataset = test_dataset[:32]
            args.num_workers = 1

        train_dataset = train_dataset[dist.get_rank()::dist.get_world_size()]
        print('Total size:%s' % (len(train_dataset)))

        transform_fn = ConfGenTaskTransformFn(n_noise_mol=args.n_noise_mol)
        train_dataset.transform(transform_fn, num_workers=args.num_workers)
        # valid_dataset.transform(transform_fn, num_workers=args.num_workers)
        # test_dataset.transform(transform_fn, num_workers=args.num_workers)

        collate_fn = ConfGenTaskCollateFn(
            atom_names=encoder_config['atom_names'],
            bond_names=encoder_config['bond_names'],
            bond_float_names=encoder_config['bond_float_names'],
            bond_angle_float_names=encoder_config['bond_angle_float_names'],
            dihedral_angle_float_names=encoder_config['dihedral_angle_float_names']
            # pretrain_tasks=head_config['pretrain_tasks']
        )

        train_data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn
        )

        # prior = GNNModel(prior_config)
        # encoder = GeoGNNModel(encoder_config)
        # decoder = GNNModel(decoder_config)

        model = VAE(
            prior_config=prior_config,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            head_config=head_config,
            n_layers=2,
            vae_beta=args.vae_beta
        )

        if not args.init_model is None and not args.init_model == "":
            # compound_encoder.set_state_dict(paddle.load(args.init_model))
            model.set_state_dict(paddle.load(args.init_model))
            print('Load state_dict from %s' % args.init_model)

        model_without_ddp = model
        args.disable_tqdm = False

        if args.distributed:
            model = paddle.DataParallel(model, find_unused_parameters=True)

        model_params = model_without_ddp.parameters()
        optimizer = paddle.optimizer.Adam(
            learning_rate=args.learning_rate,
            parameters=model_params,
            weight_decay=args.weight_decay
        )
        num_params = sum(p.numel() for p in model_params)
        print(f"#Params: {num_params}")

        train_history = []
        root = os.path.join(args.model_dir, args.dataset)
        if not os.path.exists(root):
            os.makedirs(root)

        for epoch in range(args.epochs):
            if not args.distributed or dist.get_rank() == 0:
                print("\n=====Epoch {:0>3}".format(epoch))
                print(f"Training at 0 / {dist.get_world_size()}...")
            # loss_dict = train(args, model, encoder_opt, decoder_opt, head_opt, train_data_gen, train_num)
            loss_dict = train(model, optimizer, train_data_gen, args)
            print(loss_dict)

            if not args.distributed or dist.get_rank() == 0:

                train_history.append(loss_dict)
                with open(os.path.join(root, "./train_history.pkl"), 'wb') as f:
                    pickle.dump(train_history, f)

                # print("Validating...")
                # valid_rmsd = evaluate(args, model, valid_data_gen, valid_num)
                # test_rmsd = evaluate(args, model, test_data_gen, test_num)
                # print(f"[Epoch: {epoch:0>3}] valid rmsd: {valid_rmsd}, test rmsd: {test_rmsd}")

                paddle.save(model.state_dict(), os.path.join(root, 'params', f'iter_{epoch}', 'model.pdparams'))
                # paddle.save(model.prior.state_dict(), os.path.join(root, 'params', f'iter_{epoch}', 'prior.pdparams'))
                # paddle.save(model.decoder.state_dict(), os.path.join(root, 'params', f'iter_{epoch}', 'decoder.pdparams'))


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--dataset", type=str, default='debug')
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--prior_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--head_config", type=str)
    # parser.add_argument("--aux_config", type=str)

    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default='./debug')

    # parser.add_argument("--encoder_lr", type=float, default=0.001)
    # parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--n_noise_mol", type=int, default=1)

    parser.add_argument("--vae_beta", type=float, default=1.0)

    # parser.add_argument("--recycle", type=int, default=1)
    # parser.add_argument("--num_message_passing_steps", type=int, default=3)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)

    args = parser.parse_args()

    if args.distributed:
        dist.init_parallel_env()

    main(args)


if __name__ == "__main__":
    main_cli()
