import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from miccai_learn2reg_2023.learn2reg2023.utils.train_model import Trainer
from dataset.dataloader import NLSTDataset
from torch.utils.data import DataLoader

# parser


def argParser():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=None, help="path to data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="path to output directory"
    )
    parser.add_argument(
        "--split_dir", type=str, default=None, help="path to split directory"
    )
    parser.add_argument("--model", type=str, default="LKU-Net", help="model name")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    args = parser.parse_args()
    return args


def main(args):
    # init model
    model = Trainer(args)
    # init dataset
    train_dataset = NLSTDataset(args, mode="train")
    val_dataset = NLSTDataset(args, mode="val")

    # init dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # train
    model.train(train_loader, val_loader)


if __name__ == "__main__":
    args = argParser()
    main(args)
