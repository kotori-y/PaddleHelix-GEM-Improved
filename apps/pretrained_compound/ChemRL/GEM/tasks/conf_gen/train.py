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


def get_steps_per_epoch(train_num, args):
    """tbd"""
    # add as argument
    if args.debug:
        train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    if args.distributed:
        steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
    return steps_per_epoch


def train(args, model, opt, train_data_gen, train_num):
    model.train()

    steps = get_steps_per_epoch(train_num, args)
    step = 0
    loss_accum_dict = defaultdict(float)

    for net_input, batch, prior_pos_list, gt_pos_list in train_data_gen:
        net_input = {k: v.tensor() for k, v in net_input.items()}

        gt_pos_list = paddle.to_tensor(np.vstack(gt_pos_list), dtype=paddle.float32)
        prior_pos_list = paddle.to_tensor(np.vstack(prior_pos_list), dtype=paddle.float32)

        net_input["batch"] = batch
        # net_input["gt_poses"] = gt_pos_list
        net_input["prior_poses"] = prior_pos_list

        extra_output, pos_list = model(**net_input)
        loss, loss_dict = compute_loss(extra_output, pos_list, gt_pos_list, batch, args)

        loss.backward()
        opt.step()

        # encoder_opt.clear_grad()
        # decoder_opt.clear_grad()
        # head_opt.clear_grad()
        opt.clear_grad()

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v

        step += 1
        if step > steps:
            print("jumpping out")
            break

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def evaluate(args, model, valid_data_gen, valid_num):
    model.eval()

    steps = get_steps_per_epoch(valid_num, args)
    step = 0

    mol_labels = []
    mol_preds = []

    for net_input, batch, prior_pos_list, gt_pos_list in valid_data_gen:
        net_input = {k: v.tensor() for k, v in net_input.items()}

        gt_pos_list = paddle.to_tensor(np.vstack(gt_pos_list), dtype=paddle.float32)
        prior_pos_list = paddle.to_tensor(np.vstack(prior_pos_list), dtype=paddle.float32)

        net_input["batch"] = batch
        # net_input["gt_poses"] = gt_pos_list
        net_input["prior_poses"] = prior_pos_list

        with paddle.no_grad():
            _, pos_list = model(**net_input)
        pred = pos_list[-1]

        batch_size = len(batch["mols"])
        n_nodes = batch["num_nodes"].tolist()
        pre_nodes = 0

        for i in range(batch_size):
            mol_labels.append(batch["mols"][i])
            mol_preds.append(
                set_rdmol_positions(batch["mols"][i], pred[pre_nodes: pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

        step += 1
        if step > steps:
            print("jumpping out")
            break

    rmsd_list = []
    for gen_mol, ref_mol in zip(mol_preds, mol_labels):
        try:
            rmsd_list.append(get_best_rmsd(gen_mol, ref_mol))
        except Exception as e:
            continue

    return np.mean(rmsd_list)


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

    # prior_params = prior_model.parameters()
    # encoder_params = encoder_model.parameters()
    # decoder_params = decoder_model.parameters()
    # head_params = exempt_parameters(model.parameters(), prior_params + encoder_params + decoder_params)
    #
    # encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params + prior_params)
    # decoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=decoder_params)
    # head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)

    model_params = model.parameters()
    optimizer = paddle.optimizer.Adam(args.encoder_lr, parameters=model_params)
    print('Total param num: %s' % (len(model.parameters())))

    if args.distributed:
        model = paddle.DataParallel(model)

    train_dataset = load_pickled_mol_to_dataset(args.train_dataset)
    train_num = len(train_dataset)

    valid_dataset = load_pickled_mol_to_dataset(args.valid_dataset)
    valid_num = len(valid_dataset)

    test_dataset = load_pickled_mol_to_dataset(args.test_dataset)
    test_num = len(test_dataset)

    if args.debug:
        train_dataset = train_dataset[:128]
        valid_dataset = valid_dataset[:64]
        test_dataset = test_dataset[:64]

    train_dataset = train_dataset[dist.get_rank()::dist.get_world_size()]
    print('Total size:%s' % (len(train_dataset)))

    transform_fn = ConfGenTaskTransformFn(use_self_pos=True)
    train_dataset.transform(transform_fn, num_workers=args.num_workers)
    valid_dataset.transform(transform_fn, num_workers=args.num_workers)
    test_dataset.transform(transform_fn, num_workers=args.num_workers)

    collate_fn = ConfGenTaskCollateFn(
        atom_names=compound_encoder_config['atom_names'],
        bond_names=compound_encoder_config['bond_names'],
        bond_float_names=compound_encoder_config['bond_float_names'])

    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn)

    valid_data_gen = valid_dataset.get_data_loader(
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    test_data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    train_history = []
    for epoch in range(args.max_epoch):
        if not args.distributed or dist.get_rank() == 0:
            print("\n=====Epoch {}".format(epoch))
            print("Training...")
        # loss_dict = train(args, model, encoder_opt, decoder_opt, head_opt, train_data_gen, train_num)
        loss_dict = train(args, model, optimizer, train_data_gen, train_num)

        if not args.distributed or dist.get_rank() == 0:
            train_history.append(loss_dict)
            with open("./train_history.pkl", 'wb') as f:
                pickle.dump(train_history, f)

            print("Validating...")
            valid_rmsd = evaluate(args, model, valid_data_gen, valid_num)
            test_rmsd = evaluate(args, model, test_data_gen, test_num)
            print(f"valid rmsd: {valid_rmsd}, test rmsd: {test_rmsd}")

            # paddle.save(compound_encoder.state_dict(),
            #             '%s/epoch%d/compound_encoder.pdparams' % (args.model_dir, epoch_id))
            # paddle.save(model.state_dict(),
            #             '%s/epoch%d/model.pdparams' % (args.model_dir, epoch_id))
            paddle.save(model.state_dict(), '%s/epoch%d.pdparams' % (args.model_dir, epoch))


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)

    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--valid_dataset", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)

    parser.add_argument("--cached_data_path", type=str, default=None)

    parser.add_argument("--compound_encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--vae_beta", type=float, default=1)
    parser.add_argument("--aux_loss", type=float, default=0.2)
    parser.add_argument("--num_message_passing_steps", type=int, default=3)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)

    args = parser.parse_args()

    if args.distributed:
        dist.init_parallel_env()

    main(args)


if __name__ == "__main__":
    main_cli()
