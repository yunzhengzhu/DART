#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils.train_model import Trainer
from dataset.dataloader import NLSTDataset
from dataset.data_utils import torch2torchiodataset, augmentations
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
        "--exp_name",
        type=str,
        default=None,
        help="customize save name",
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
        "--orient_stand",
        action="store_true",
        default=False,
        help="orientation standardization on case 208, 260, and 298",
    )
    parser.add_argument(
        "--mask_dir", type=str, default=None, help="specify customized mask dir"
    )
    parser.add_argument(
        "--organs",
        nargs="+",
        help="list of organs",
        default=None,
        type=str,
    ) 
    parser.add_argument(
        "--side",
        nargs="+",
        help="list of mask sides (left, right)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--specific_regions",
        nargs="+",
        help="list of specific regions",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--texture_mask_dir", type=str, default=None, help="specify texture mask dir"
    )

    parser.add_argument(
        "--random_sample", type=int, default=None, help="# of randomly sampled kp"
    )
    parser.add_argument(
        "--kp_dir", type=str, default=None, help="specify customized kp dir"
    )
    parser.add_argument(
        "--affine_aug",
        type=str,
        default=None,
        help="affine augmentation on images",
    )
    parser.add_argument(
        "--affine_param",
        type=float,
        default=0.035,
        help="affine transformation parameter",
    )
    parser.add_argument(
        "--flip_aug",
        type=str,
        default=None,
        help="flip augmentation on images",
    )
    parser.add_argument(
        "--flip_axis",
        nargs="+",
        help="flip axises",
        default=[1, -1],
        type=int,
    )
    parser.add_argument(
        "--kp_aug",
        action="store_true",
        default=False,
        help="keypoint augmentation",
    )
    parser.add_argument(
        "--foerstner_kernel",
        type=int,
        default=9,
        help="kernel for foerstner operator",
    )
    parser.add_argument(
        "--foerstner_points",
        type=int,
        default=2000,
        help="number of foerstner points",
    )
    parser.add_argument(
        "--foerstner_thres",
        type=float,
        default=0.03,
        help="threshold for points matching",
    )

    parser.add_argument(
        "--augs", 
        nargs="+",
        help="list of augs",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--transform_type", type=str, default="same", help="same/diff on mov and fixed"
    )

    # feature extraction
    parser.add_argument(
        "--mind_feature",
        action="store_true",
        default=False,
        help="extract mind features for img",
    )
    parser.add_argument(
        "--masked_img",
        action="store_true",
        default=False,
        help="extract masked region only from img",
    )

    # model
    parser.add_argument("--model_type", type=str, default="LKU-Net", help="model name")
    parser.add_argument(
        "--start_channel", type=int, default=8, help="start channel U-Net"
    )
    parser.add_argument("--blur_factor", type=int, default=None, help="disp blurring")
    parser.add_argument(
        "--diff", action="store_true", default=False, help="use DiffeomorphicTransform"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="path to pretrained model"
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
    parser.add_argument(
        "--use_scaler", action="store_true", default=False, help="use gradient scaler"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--max_epoch",
        type=float,
        default=100.0,
        help="maximum number of epochs for scheduler",
    )
    parser.add_argument("--lrf", type=float, default=None, help="learning rate factor")
    parser.add_argument(
        "--milestones",
        nargs="+",
        help="list of scheduler milestone epochs",
        default=[10.0],
        type=float,
    )
    parser.add_argument(
        "--es", action="store_true", default=False, help="early stopping"
    )
    parser.add_argument(
        "--es_criterion", type=str, default="total", help="early stopping criterion"
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

    if args.es_criterion not in args.loss and args.es_criterion != "total":
        raise ValueError("Early stopping criterion not in loss!")

    # continue training on previous checkpoint
    if args.continue_training:
        # load args json
        with open(os.path.join(args.exp_dir, "args.json"), "r") as f:
            prev_args = json.load(f)

        # update args
        for key, value in prev_args.items():
            if key not in ["continue_training", "exp_dir", "epochs", "sche"]:
                setattr(args, key, value)

        # init model
        model = Trainer(args, mode="train")
        # load checkpoint
        model.load_prev()

    else:
        # create experiment folder
        if args.exp_name is not None:
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
            
            if args.masked_img:
                exp_name += "_usemaskedimg"

            if args.mask_dir:
                exp_name += f"_{args.mask_dir}_{args.organs}_{args.side}_{args.specific_regions}"

            if args.random_sample:
                exp_name += f"_rs{args.random_sample}"

            if args.kp_dir:
                exp_name += f"_{args.kp_dir}"

            if args.texture_mask_dir:
                exp_name += f"_{args.texture_mask_dir}"

            if args.affine_aug:
                exp_name += f"_affineaug{args.affine_aug}_param{args.affine_param}"
            
            if args.flip_aug:
                exp_name += f"_flipaug{args.flip_aug}_param{args.flip_axis}"

            if args.kp_aug:
                exp_name += f"_kpaug{args.kp_aug}_k{args.foerstner_kernel}_np{args.foerstner_points}_t{args.foerstner_thres}"
           
            if args.augs:
                exp_name += f"_aug_{'_'.join([aug for aug in args.augs])}_trans{args.transform_type}"


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
    if args.mask_dir:
        mask_info = {
            "organs": args.organs,
            "side": args.side,
            "specific_regions": args.specific_regions,
        }
    else:
        mask_info = {}
    if args.kp_aug:
        kp_aug_info = {
            "kernel": args.foerstner_kernel,
            "num_points": args.foerstner_points,
            "threshold": args.foerstner_thres,
        }
    else:
        kp_aug_info = {}
    
    train_dataset = NLSTDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        mode="train",
        downsample=args.downsample,
        preprocess=args.preprocess,
        orient_stand=args.orient_stand,
        random_sample=args.random_sample,
        kp_dir=args.kp_dir,
        mask_dir=args.mask_dir,
        mask_info=mask_info,
        affine_aug=args.affine_aug,
        affine_prob=0.5,
        affine_param=args.affine_param,
        flip_aug=args.flip_aug,
        flip_prob=0.5,
        flip_axis=args.flip_axis,
        kp_aug=args.kp_aug,
        kp_aug_info=kp_aug_info,
        texture_mask_dir=args.texture_mask_dir,
    )
    
    if args.augs:
        print(f"Transferring to TorchIO dataset for {args.augs} augs w. {args.transform_type} type...")
        aug_process = augmentations(args.augs, p=1)
        # Use TorchIO based subject dataset to apply packaged augmentation
        train_dataset = torch2torchiodataset(train_dataset, aug_process, transform=args.transform_type, downsample=args.downsample)
        print("done")
    
    # init dataloader
    if args.transform_type == "diff" and len(train_dataset) == 2:
        # shuffle should be true to make sure number of keypoints for fixed and moving are the same
        train_loader_f = DataLoader(train_dataset[0], batch_size=args.batch_size, shuffle=False, num_workers=1, worker_init_fn=seed_worker, pin_memory=True)
        train_loader_m = DataLoader(train_dataset[1], batch_size=args.batch_size, shuffle=False, num_workers=1, worker_init_fn=seed_worker, pin_memory=True)
        train_loader = list(zip(train_loader_f, train_loader_m))
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=1,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
    
    val_dataset = NLSTDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        mode="val",
        downsample=args.downsample,
        preprocess=args.preprocess,
        random_sample=args.random_sample,
        eval_with_mask=args.eval_with_mask,
        mask_dir=args.mask_dir,
        mask_info=mask_info,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # train
    results = model.train(train_loader, val_loader)

    results.to_csv(os.path.join(args.exp_dir, f"results.csv"))


if __name__ == "__main__":
    args = argParser()
    main(args)
