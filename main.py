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
    parser.add_argument("--model_type", type=str, default="LKU-Net", help="model name")
    parser.add_argument(
        "--loss", type=str, default="NCC", help="similarity loss function"
    )
    parser.add_argument(
        "--smooth_w", type=float, default=0.1, help="smooth loss weight"
    )
    parser.add_argument("--opt", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    parser.add_argument(
        "--es_warmup", type=int, default=10, help="early stopping warmup"
    )
    parser.add_argument(
        "--es_tolerence", type=int, default=20, help="early stopping tolerence"
    )
    args = parser.parse_args()
    return args


def main(args):
    # create experiment folder
    exp_name = f"{args.model_type}_{args.loss}_{args.opt}_lr{args.lr}_bs{args.batch_size}_seed{args.seed}"
    exp_dir = os.path.join(args.result_dir, exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    args.exp_dir = exp_dir
    # save args json
    with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # init model
    model = Trainer(args)
    # init dataset
    train_dataset = NLSTDataset(
        data_dir=args.data_dir, json_file=args.json_file, mode="train"
    )
    val_dataset = NLSTDataset(
        data_dir=args.data_dir, json_file=args.json_file, mode="val"
    )

    # init dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # train
    results = model.train(train_loader, val_loader)

    results.to_csv(os.path.join(args.exp_dir, "results.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
