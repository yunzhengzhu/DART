#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils.train_model import Trainer
from dataset.dataloader import NLSTDataset
from torch.utils.data import DataLoader
import json


def argParser():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="path to experiment directory",
    )
    parser.add_argument(
        "--save_df",
        action="store_true",
        default=False,
        help="save displacement field",
    )
    args = parser.parse_args()
    return args


def main(args):
    # set up arguments
    with open(os.path.join(args.exp_dir, "args.json"), "r") as file:
        loaded_args = json.load(file)
    for key, value in loaded_args.items():
        if key not in ["exp_dir", "save_df"]:
            setattr(args, key, value)

    # create save directory
    args.save_dir = os.path.join(args.exp_dir, "eval")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # init model
    model = Trainer(args, mode="eval")

    # init dataset
    val_dataset = NLSTDataset(
        data_dir=args.data_dir, json_file=args.json_file, mode="val", downsample=args.downsample, preprocess=args.preprocess,
    )

    # init dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train
    results = model.predict(val_loader)
    results.to_csv(os.path.join(args.save_dir, "results.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
