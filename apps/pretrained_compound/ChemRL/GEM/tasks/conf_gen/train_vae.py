import argparse
import copy
import pickle
from collections import defaultdict

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.optimizer.lr import LambdaDecay

from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.vae import VAE
from datasets import load_pickled_mol_to_dataset, ConfGenTaskTransformFn, ConfGenTaskCollateFn
from pahelix.utils import load_json_config
from pahelix.model_zoo.gem_model import GeoGNNModel, GNNModel


# from loss.e3_loss import compute_loss
# # from model.gnn import ConfGenModel
# from pahelix.utils import load_json_config
# from utils import exempt_parameters, set_rdmol_positions, get_best_rmsd, WarmCosine
#
# from pahelix.model_zoo.gem_model import GeoGNNModel, GeoPredModel

def main(args):
    prior_config = load_json_config(args.prior_config)
    encoder_config = load_json_config(args.encoder_config)
    decoder_config = load_json_config(args.decoder_config)
    # aux_config = load_json_config(args.aux_config)
    head_config = load_json_config(args.head_config)
    #
    # if args.dropout_rate is not None:
    #     encoder_config['dropout_rate'] = args.dropout_rate
    #     decoder_config['dropout_rate'] = args.dropout_rate
    #     # down_config['dropout_rate'] = args.dropout_rate

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
            args.num_workers = 1

        train_dataset = train_dataset[dist.get_rank()::dist.get_world_size()]
        print('Total size:%s' % (len(train_dataset)))

        transform_fn = ConfGenTaskTransformFn(n_noise_mol=5)
        train_dataset.transform(transform_fn, num_workers=args.num_workers)
        valid_dataset.transform(transform_fn, num_workers=args.num_workers)
        test_dataset.transform(transform_fn, num_workers=args.num_workers)

        collate_fn = ConfGenTaskCollateFn(
            atom_names=encoder_config['atom_names'],
            bond_names=encoder_config['bond_names'],
            bond_float_names=encoder_config['bond_float_names'],
            bond_angle_float_names=encoder_config['bond_angle_float_names'],
            dihedral_angle_float_names=encoder_config['dihedral_angle_float_names']
        )

        train_data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            collate_fn=collate_fn
        )

        prior = GNNModel(prior_config)
        encoder = GeoGNNModel(encoder_config)

        model = VAE(prior, encoder, head_config)

        for prior_graph, encoder_graph in train_data_gen:

            for k in prior_graph:
                prior_graph[k] = prior_graph[k].tensor()

            for k in encoder_graph:
                encoder_graph[k] = encoder_graph[k].tensor()

            model(prior_graph, encoder_graph)


def main_cli():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)

    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--valid_dataset", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)

    parser.add_argument("--cached_data_path", type=str, default=None)

    parser.add_argument("--prior_config", type=str)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--decoder_config", type=str)
    parser.add_argument("--head_config", type=str)
    # parser.add_argument("--aux_config", type=str)

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
