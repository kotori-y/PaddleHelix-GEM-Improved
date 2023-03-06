import argparse
import copy
import pickle
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LambdaDecay

from datasets import load_sdf_mol_to_dataset, load_pickled_mol_to_dataset, ConfGenTaskTransformFn, ConfGenTaskCollateFn
from loss.e3_loss import compute_loss
from model.gnn import ConfGenModel
from pahelix.utils import load_json_config
from utils import exempt_parameters, set_rdmol_positions, get_best_rmsd, WarmCosine

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
    print(f"lr: {opt.get_lr()}")

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
        opt.clear_grad()
        # scheduler.step()

        for k, v in loss_dict.items():
            loss_accum_dict[k] += v
        # for k in loss_accum_dict.keys():
        #     print(f"{k} : {loss_accum_dict[k] / step + 1}")

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
    encoder_config = load_json_config(args.encoder_config)
    decoder_config = load_json_config(args.decoder_config)
    down_config = load_json_config(args.down_config)

    if args.dropout_rate is not None:
        encoder_config['dropout_rate'] = args.dropout_rate
        decoder_config['dropout_rate'] = args.dropout_rate
        # down_config['dropout_rate'] = args.dropout_rate

    if not args.distributed or dist.get_rank() == 0:
        print("===> Converting data...")

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

    transform_fn = ConfGenTaskTransformFn(is_inference=False, add_noise=True, only_atom_bond=True)
    train_dataset.transform(transform_fn, num_workers=args.num_workers)
    valid_dataset.transform(transform_fn, num_workers=args.num_workers)
    test_dataset.transform(transform_fn, num_workers=args.num_workers)

    collate_fn = ConfGenTaskCollateFn(
        atom_names=encoder_config['atom_names'],
        bond_names=encoder_config['bond_names'],
        bond_float_names=encoder_config['bond_float_names'],
        # bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
        # dihedral_angle_float_names=compound_encoder_config['dihedral_angle_float_names']
    )

    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    valid_data_gen = valid_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    test_data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn)

    model = ConfGenModel(encoder_config=encoder_config, decoder_config=decoder_config, down_config=down_config,
                         recycle=args.recycle)

    model_params = model.parameters()
    # lrscheduler = WarmCosine(tmax=len(train_data_gen) * args.period, warmup=int(4e3))
    # scheduler = LambdaDecay(args.encoder_lr, lr_lambda=lambda x: lrscheduler.step(x))
    optimizer = paddle.optimizer.Adam(learning_rate=args.encoder_lr, parameters=model_params)
    print('Total param num: %s' % (len(model.parameters())))

    if args.distributed:
        model = paddle.DataParallel(model)

    train_history = []
    for epoch in range(args.max_epoch):
        if not args.distributed or dist.get_rank() == 0:
            print("\n=====Epoch {}".format(epoch))
            print(f"Training at 0 / {dist.get_world_size()}...")
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

    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--down_config", type=str)
    parser.add_argument("--init_model", type=str, default=None)
    parser.add_argument("--model_dir", type=str)

    parser.add_argument("--encoder_lr", type=float, default=0.001)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--head_lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    parser.add_argument("--vae_beta", type=float, default=1)
    parser.add_argument("--aux_loss", type=float, default=0.2)
    parser.add_argument("--ang_lam", type=float, default=0.2)
    parser.add_argument("--bond_lam", type=float, default=0.2)

    parser.add_argument("--recycle", type=int, default=1)
    parser.add_argument("--num_message_passing_steps", type=int, default=3)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)

    args = parser.parse_args()

    if args.distributed:
        dist.init_parallel_env()

    main(args)


if __name__ == "__main__":
    main_cli()
