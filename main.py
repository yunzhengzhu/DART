#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils.train_model import Trainer
from dataset.dataloader import NLSTDataset
from torch.utils.data import DataLoader
import json
from utils.train_utils import set_seed, seed_worker


def argParser():
    parser = ArgumentParser()
    # data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/databases/imgreg/NLST2023",
        help="path to data directory",
    )
    parser.add_argument(
        "--json_file", type=str, default="NLST_dataset.json", help="name of json file"
    )
    parser.add_argument(
        "--result_dir", type=str, default="./results", help="path to output directory"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None, help="customize save name",
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="downsample factor on all dims"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="preprocess images",
    )    
    parser.add_argument(
        "--random_sample", type=int, default=99999, help="# of randomly sampled kp"
    )  

    # feature extraction
    parser.add_argument(
        "--mind_feature", action="store_true", default=False, help="extract mind features for img"
    )

    # model
    parser.add_argument("--model_type", type=str, default="LKU-Net", help="model name")
    parser.add_argument(
        "--start_channel", type=int, default=8, help="start channel U-Net"
    )
    parser.add_argument(
        "--blur_factor", type=int, default=None, help="disp blurring")
    parser.add_argument(
        "--diff", action="store_true", default=False, help="use DiffeomorphicTransform"
    )
    parser.add_argument(
        "--pretrained",type=str, default=None, help="path to pretrained model"
    )
    parser.add_argument(
        "--freeze", type=str, default=None, help="freezing module in network"
    )

    # training
    parser.add_argument(
        "--loss",
        nargs="+",
        help="list of loss",
        default=["TRE"],
        type=str,
    )
    parser.add_argument(
        "--loss_weight",
        nargs="+",
        help="list of loss weight",
        default=[1.0, 0.1],
        type=float,
    )
    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--sche", type=str, default=None, help="scheduler")
    parser.add_argument("--use_scaler", action="store_true", default=False, help="use gradient scaler")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=100,
        help="maximum number of epochs for scheduler",
    )
    parser.add_argument("--lrf", type=float, default=None, help="learning rate factor")
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    parser.add_argument(
        "--es_criterion", type=str, default='total', help="early stopping criterion"
    )
    parser.add_argument(
        "--es_warmup", type=int, default=0, help="early stopping warmup"
    )
    parser.add_argument(
        "--es_patience", type=int, default=20, help="early stopping patience"
    )
    parser.add_argument(
        "--log", action="store_true", default=False, help="log training"
    )
    parser.add_argument(
        "--print_every", type=int, default=10, help="print every n batches"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--save_df",
        action="store_true",
        default=False,
        help="save displacement field",
    )
    parser.add_argument(
        "--save_warped",
        action="store_true",
        default=False,
        help="save warped image",
    )

    # validation
    parser.add_argument(
        "--rev_metric", action="store_true", default=False, help="track reverse metrics"
    )

    # continue training if your model is interrupted
    parser.add_argument(
        "--continue_training",
        action="store_true",
        default=False,
        help="continue training",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="path to checkpoint",
    )
    args = parser.parse_args()
    return args


def main(args):
    # set seed
    set_seed(args.seed)

    if args.es_criterion not in args.loss and args.es_criterion != 'total':
        raise ValueError("Early stopping criterion not in loss!")

    # continue training on previous checkpoint
    if args.continue_training:
        # load args json
        with open(os.path.join(args.exp_dir, "args.json"), "r") as f:
            prev_args = json.load(f)

        # update args
        for key, value in prev_args.items():
            if key not in ["continue_training", "exp_dir", "epochs"]:
                setattr(args, key, value)

        # init model
        model = Trainer(args, mode="train")
        # load checkpoint
        model.load_prev()

    else:
        # create experiment folder
        if args.exp_name != None:
            exp_name = args.exp_name
        else:
            if args.sche:
                exp_name = f"{args.model_type}_{args.start_channel}_{'_'.join([l+str(lw) for l, lw in zip (args.loss,args.loss_weight)])}_{args.opt}_lr{args.lr}_sche{args.sche}_lrf{args.lrf}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
            else:
                exp_name = f"{args.model_type}_{args.start_channel}_{'_'.join([l+str(lw) for l, lw in zip (args.loss,args.loss_weight)])}_{args.opt}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
            
            if args.freeze:
                exp_name += f"_freeze{args.freeze}"

            if args.use_scaler:
                exp_name += "_scaler"

            if args.blur_factor:
                exp_name += f"_blur{args.blur_factor}"

        if args.diff:
            exp_name += "_difftrans"
        
        if args.preprocess:
            exp_name += "_preprocess"

        if args.mind_feature:
            exp_name += "_usemind"

        if args.random_sample != 99999:
            exp_name += f'_rs{args.random_sample}'

        exp_dir = os.path.join(args.result_dir, exp_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        else:
            raise ValueError("Experiment folder already exists!")
        args.exp_dir = exp_dir
        # save args json
        with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        # init model
        model = Trainer(args, mode="train")
    # init dataset
    train_dataset = NLSTDataset(
        data_dir=args.data_dir, json_file=args.json_file, mode="train", downsample=args.downsample, preprocess=args.preprocess, random_sample=args.random_sample
    )
    val_dataset = NLSTDataset(
        data_dir=args.data_dir, json_file=args.json_file, mode="val", downsample=args.downsample, preprocess = args.preprocess, random_sample=args.random_sample
    )

    # init dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train
    results = model.train(train_loader, val_loader)

    results.to_csv(os.path.join(args.exp_dir, "results.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
