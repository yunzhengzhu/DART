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
    parser.add_argument(
        "--save_warped",
        action="store_true",
        default=False,
        help="save warped image",
    )
    parser.add_argument(
        "--eval_diff",
        action="store_true",
        default=False,
        help="evaluation with diffeomorphism",
    )
    parser.add_argument(
        "--eval_blur_factor",
        type=int,
        default=None,
        help="evaluation with blurring",
    )
    parser.add_argument(
        "--eval_with_mask",
        action="store_true",
        default=False,
        help="evaluation with predefined masks in training",
    )
    parser.add_argument(
        "--nodule_kp_dir",
        type=str,
        default=None,
        help="path to nodule detection keypoints",
    )
    parser.add_argument(
        "--nodule_id",
        type=str,
        default=None,
        help="id of nodule",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="nodule_kpt",
        help="lm to compute tre (nodule_kpt, nodule_center)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        help="specify the evaluation mode val/test (default: val)"
    )
    args = parser.parse_args()
    return args


def main(args):
    # set up arguments
    with open(os.path.join(args.exp_dir, "args.json"), "r") as file:
        loaded_args = json.load(file)
    for key, value in loaded_args.items():
        if key not in ["exp_dir", "save_df", "save_warped", "nodule_kp_dir"]:
            setattr(args, key, value)

    # create save directory
    args.save_dir = os.path.join(args.exp_dir, args.mode)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # init model
    model = Trainer(args, mode=args.mode)

    # init dataset
    if args.eval_with_mask:
        mask_dir = args.mask_dir
        mask_info = {
            "organs": args.organs,
            "side": args.side,
            "specific_regions": args.specific_regions,
        }
    else:
        mask_dir = None
        mask_info = {}
    
    eval_dataset = NLSTDataset(
        data_dir=args.data_dir, 
        json_file=args.json_file, 
        mode=args.mode, 
        downsample=args.downsample, 
        preprocess=args.preprocess,
        eval_with_mask=args.eval_with_mask,
        mask_dir=mask_dir,
        mask_info=mask_info,
        nodule_kp_dir=args.nodule_kp_dir,
        nodule_id=args.nodule_id,
        lm=args.lm,
    )

    # init dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # train
    if args.mode == "val" or args.mode == "test":
        results = model.predict(eval_loader)
        if args.nodule_kp_dir:
            if args.lm == "nodule_kpt":
                results.to_csv(os.path.join(args.save_dir, f"results_{args.mode}_{args.lm}_{os.path.splitext(args.nodule_id)[0]}.csv"))
            else:
                results.to_csv(os.path.join(args.save_dir, f"results_{args.mode}_{args.lm}.csv"))
        else:
            results.to_csv(os.path.join(args.save_dir, f"results_{args.mode}.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
