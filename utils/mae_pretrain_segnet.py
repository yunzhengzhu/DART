from typing import Union, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import numpy as np
import pandas as pd
import os
import random
import time
import math
import nibabel as nib
from argparse import ArgumentParser
from model.mae_pretrain_segnet import MAE_Pretrain_SegNet
from utils.loss_utils import smoothLoss, NCC, GNCC, Dice, MSE, SAD, TRE, MINDSSC, Seg_MSE
from utils.train_utils import EarlyStopping
from utils.metric_utils import jacobian_determinant, compute_tre, compute_dice
from utils.feature_utils import mindssc
from tensorboardX import SummaryWriter

class baseTrainer:
    def __init__(self, args: ArgumentParser, mode: str = "train") -> None:
        self.args = args
        self.downsample = args.downsample
        self.use_scaler = args.use_scaler
        self.opt = args.opt
        self.lr = args.lr
        self.sche = args.sche
        if self.sche:
            self.sche_param = {
                "cosine": {"max_epoch": args.max_epoch},
                "lambdacosine": {"max_epoch": args.max_epoch, "lr_factor": args.lrf},
                "multisteplambdacosine": {"max_epoch": args.max_epoch, "lr_factor": args.lrf, "milestones": args.milestones}
            }
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.loss = args.loss
        self.loss_weight = args.loss_weight
        self.model_type = args.model_type
        self.es = args.es
        self.es_warmup = args.es_warmup
        self.es_patience = args.es_patience
        self.start_channel = args.start_channel
        self.exp_dir = args.exp_dir
        self.log = args.log
        self.print_every = args.print_every
        self.freeze = args.freeze
        self.pretrained = args.pretrained if mode == "train" else None
        self.rev_metric = args.rev_metric
        self.es_criterion = args.es_criterion
        self.mind_feature = args.mind_feature
        self.masked_img = args.masked_img
        if args.organs:
            self.num_masks = len(args.organs) 
        elif args.specific_regions:
            self.num_masks = len(args.specific_regions)
        else:
            self.num_masks = 1
        self.transform_type = args.transform_type
        self.use_augs = True if args.augs != None else False
        self.use_texture_mask = True if args.texture_mask_dir else False
        self.total_epochs = args.epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.__init_model()
        self.__init_loss()
        self.__init_optimizer()
        self.__init_scheduler()
        self.__init_scaler()
        self.__init_logger()
        self.__init_es()

    def __init_scaler(self):
        if self.use_scaler:
            print(f"Initiate gradient scaler", end=" ")
            self.scaler = torch.cuda.amp.GradScaler()
            print("...done")
        else:
            self.scaler = None

    def __init_optimizer(self):
        print(f"Initiate {self.opt} optimizer", end=" ")

        if self.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        print("...done")

    def __init_scheduler(self):
        if self.sche != None:
            print(f"Initiate {self.sche} scheduler", end=" ")
            if self.sche == "cosine":
                sche_param = self.sche_param["cosine"]
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=sche_param["max_epoch"], eta_min=0
                )
            elif self.sche == "lambdacosine":
                sche_param = self.sche_param["lambdacosine"]
                lf = (
                    lambda x: (
                        (1 + math.cos(x * math.pi / sche_param["max_epoch"])) / 2
                    )
                    * (1 - sche_param["lr_factor"])
                    + sche_param["lr_factor"]
                )
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lf
                )
            elif self.sche == "multisteplambdacosine":
                sche_param = self.sche_param["multisteplambdacosine"]
                def lambda_func(x, firststep_epochs, milestone, lrf):
                    if x <= milestone:
                        coef = ((1 + math.cos(x * math.pi / firststep_epochs)) / 2) * (1 - lrf) + lrf
                    else:
                        coef = ((1 + math.cos(milestone * math.pi / firststep_epochs)) / 2) * (1 - lrf) + lrf
                    return coef

                lf = lambda x: lambda_func(x, sche_param["max_epoch"], sche_param["milestones"][0], sche_param["lr_factor"])
                
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lf
                )
            else:
                raise NotImplementedError
            print("...done")
        else:
            self.scheduler = None

    def __init_loss(self):
        print(f"Initiate {self.loss} loss with weight {self.loss_weight}", end=" ")
        # self.loss_weight = self.loss_weight / np.sum(self.loss_weight)
        assert len(self.loss) == len(
            self.loss_weight
        ), "Loss and loss weight must have the same length"
        self.loss_fn = {}
        for l, lw in zip(self.loss, self.loss_weight):
            if l == "NCC":
                self.loss_fn[l] = [NCC(), lw]
            elif l == "GNCC":
                self.loss_fn[l] = [GNCC(), lw]
            elif l == "MSE":
                self.loss_fn[l] = [MSE(), lw]
            elif l == "SAD":
                self.loss_fn[l] = [SAD(), lw]
            elif l == "MINDSSC":
                self.loss_fn[l] = [MINDSSC(), lw]
            elif l == "MINDSSC_NCC":
                self.loss_fn[l] = [MINDSSC(loss_type="ncc"), lw]
            elif l == "Seg_MSE":
                self.loss_fn[l] = [Seg_MSE(), lw]
            else:
                raise NotImplementedError
        print("...done")

    def __init_model(self):
        print(f"Initiate {self.model_type} model", end=" ")
        if self.mind_feature:
            in_channel = 12
        else:
            in_channel = 1

        down_factor = 2
        if self.model_type == "MAE_ViT_Seg":
            self.model = MAE_Pretrain_SegNet(
                image_size=(224//self.downsample, 192//self.downsample, 224//self.downsample),
                in_channels=in_channel,
                num_masks=self.num_masks,
                down_factor=down_factor
            ) 
            self.mask_down_factor = 0
            print(self.model)
        else:
            raise NotImplementedError
            
        print("...done")

        if self.pretrained:
            print("Load pretrained model", end=" ")
            self.model.load_state_dict(torch.load(self.pretrained, map_location='cuda:0')["model"])
            print("...done")
        
        if self.freeze:
            print(f"Freezing {self.freeze}")
            for name, param in self.model.named_parameters():
                if self.freeze in name:
                    param.requires_grad = False

        self.model.to(self.device)

    def __init_logger(self):
        if self.log:
            print("Initiate tensorboard logger", end=" ")
            self.log_dir = os.path.join(self.exp_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print("...done")
        else:
            self.writer = None

    def __init_es(self):
        if self.es:
            print(
                "Initiate early stopping with warmup: {} and patience: {}".format(
                    self.es_warmup, self.es_patience
                ),
                end=" ",
            )
            self.early_stopping = EarlyStopping(
                warmup=self.es_warmup, patience=self.es_patience, verbose=True
            )
            print("...done")
        else:
            self.early_stopping = None


class Trainer(baseTrainer):
    def __init__(self, args, mode: str = "train") -> None:
        super(Trainer, self).__init__(args, mode=mode)
        self.start_epoch = 0
        self.epochs = args.epochs
        self.ckpt_path = os.path.join(args.exp_dir, "checkpoint.pth.tar")
        self.es_ckpt_path = os.path.join(args.exp_dir, "es_checkpoint.pth.tar")
        self.save_df = args.save_df
        self.save_warped = args.save_warped
    
    def train(
        self,
        train_loader: Union[torch.utils.data.DataLoader, Iterable[torch.utils.data.DataLoader]],
        val_loader: torch.utils.data.DataLoader,
    ) -> pd.DataFrame:
        # loop through epochs
        for i in range(self.start_epoch, self.epochs):
            self.model.train()
            print(f"------------Epoch {i+1}/{self.epochs}------------")
            # training
            train_loss_sum = 0
            train_sub_loss_sum = {l: 0.0 for l in self.loss}
            start_time = time.time()
            for batch_idx, batch_data in enumerate(train_loader):
                if self.use_augs:
                    img = batch_data['img'][tio.DATA]
                    mask = batch_data['mask'][tio.DATA]
                    multiple_mask = batch_data['mulmask'][tio.DATA]
                    kp = batch_data['kpt']
                    mask_labels = batch_data['labels'][0][0] 
                else:
                    img, kp, mask, multiple_mask, mask_labels = batch_data
                
                img = img.float().to(self.device)
                kp = kp.float().to(self.device)
                mask = mask.float().to(self.device)
                multiple_mask = multiple_mask.float().to(self.device)

                # mind feature
                if self.masked_img:
                    img = img * mask

                if self.mind_feature:
                    mind = mindssc(img)
                    model_input = mind
                else:
                    model_input = img                
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                        train_loss, train_all_loss = self.__compute_loss(
                            self.loss_fn,
                            recon_patch,
                            orig_patch,
                            img,
                            multiple_mask,
                            dec_mask,
                            downsample=self.downsample,
                        )
                    
                    self.scaler.scale(train_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                    train_loss, train_all_loss = self.__compute_loss(
                        self.loss_fn,
                        recon_patch,
                        orig_patch,
                        img,
                        multiple_mask,
                        dec_mask,
                        downsample=self.downsample,
                    )
                    train_loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                train_loss_sum += train_loss.item()
                for l, loss_value in train_sub_loss_sum.items():
                    train_sub_loss_sum[l] = loss_value + train_all_loss[l]

                # training progess batch and loss
                if batch_idx % self.print_every == 0:
                    print(
                        "Batch {} - Train Loss: {:.6f}".format(
                            batch_idx, train_loss.item()
                        )
                    )
                    for l in self.loss:
                        print("\t\t |- {} loss: {:.6f}".format(l, train_all_loss[l]))

            # update scheduler
            if self.scheduler:
                self.scheduler.step()
                scheduler_state_dict = self.scheduler.state_dict()
            else:
                scheduler_state_dict = {}
            
            train_loss_mean = train_loss_sum / len(train_loader)
            train_sub_loss_mean = {
                l: loss_value / len(train_loader)
                for l, loss_value in train_sub_loss_sum.items()
            }

            print("Epoch {} - Train Loss: {:.6f}".format(i, train_loss_mean))
            print(
                "Epoch {} - Train Loss: {:.6f} - LR: {:.4e}".format(
                    i, train_loss_mean, self.optimizer.param_groups[0]["lr"]
                )
            )
            for l, tlm in train_sub_loss_mean.items():
                print("\t\t |- {} loss: {:.6f}".format(l, tlm))

            # log training
            if self.writer:
                self.writer.add_scalar("train/Total_loss", train_loss_mean, i)
                self.writer.add_scalar(
                    "train/LR", self.optimizer.param_groups[0]["lr"], i
                )
                for l, tlm in train_sub_loss_mean.items():
                    self.writer.add_scalar("train/{}_loss".format(l), tlm, i)

            # validation
            earlystop = self.__eval(i, val_loader)

            # save model for each epoch
            state = {
                "epoch": i + 1,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": scheduler_state_dict,
                "early_stopping": self.early_stopping,
                #"logger": self.writer
            }

            torch.save(state, self.ckpt_path)

            if earlystop:
                break

        print("Finished Training...")
        results = self.predict(val_loader)

        if self.writer:
            self.writer.close()

        return results

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader) -> bool:
        val_loss_sum = 0
        val_sub_loss_sum = {l: 0.0 for l in self.loss if l != "Smooth"}
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (
                img,
                kp,
                mask,
                multiple_mask,
                mask_labels,
            ) in enumerate(val_loader):
                img = img.float().to(self.device)
                kp = kp.float().to(self.device)
                mask = mask.float().to(self.device)
                multiple_mask = multiple_mask.float().to(self.device)
                
                # mind feature
                if self.masked_img:
                    img = img * mask

                if self.mind_feature:
                    mind = mindssc(img)
                    model_input = mind
                else:
                    model_input = img
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                        
                        val_loss, val_all_loss = self.__compute_loss(
                            self.loss_fn,
                            recon_patch,
                            orig_patch,
                            img,
                            multiple_mask,
                            dec_mask,
                            downsample=self.downsample,
                        )

                else:
                    dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                    val_loss, val_all_loss = self.__compute_loss(
                        self.loss_fn,
                        recon_patch,
                        orig_patch,
                        img,
                        multiple_mask,
                        dec_mask,
                        downsample=self.downsample,
                    )
                
                val_loss_sum += val_loss.item()
                for l, loss_value in val_sub_loss_sum.items():
                    val_sub_loss_sum[l] = loss_value + val_all_loss[l]

        val_loss_mean = val_loss_sum / len(val_loader)
        val_sub_loss_mean = {
            l: loss_value / len(val_loader)
            for l, loss_value in val_sub_loss_sum.items()
        }

        print("Epoch {} - Validation Loss : {:.6f}".format(cur, val_loss_mean))
        for l, vslm in val_sub_loss_mean.items():
            print("\t\t |- {} loss: {:.6f}".format(l, vslm))

        if self.writer:
            self.writer.add_scalar("val/Total_loss", val_loss_mean, cur)
            for l, vslm in val_sub_loss_mean.items():
                self.writer.add_scalar("val/{}_loss".format(l), vslm, cur)

        # early stopping
        if self.es:
            if self.es_criterion == 'total':
                self.early_stopping(
                    epoch=cur,
                    val_loss=val_loss_mean,
                    model=self.model,
                    ckpt_path=self.es_ckpt_path,
                )
            else:
                self.early_stopping(
                    epoch=cur,
                    val_loss=val_sub_loss_mean[self.es_criterion],
                    model=self.model,
                    ckpt_path=self.es_ckpt_path,
                )   
            return self.early_stopping.early_stop
        else:
            return False

    def predict(self, val_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
        # load final model
        if self.es:
            ckpt_path = self.es_ckpt_path
        else:
            ckpt_path = self.ckpt_path

        print("=== Load model from {} ===".format(ckpt_path))
        self.model.load_state_dict(torch.load(ckpt_path)["model"])
        self.model.eval()

        val_loss_sum = 0
        val_sub_loss_sum = {l: 0.0 for l in self.loss if l != "Smooth"}
        with torch.no_grad():
            for batch_idx, (
                img,
                kp,
                mask,
                multiple_mask,
                mask_labels,
            ) in enumerate(val_loader):
                img = img.float().to(self.device)
                kp = kp.float().to(self.device)
                mask = mask.float().to(self.device)
                multiple_mask = multiple_mask.float().to(self.device)
                
                # mind feature
                if self.masked_img:
                    img = img * mask

                if self.mind_feature:
                    mind = mindssc(img)
                    model_input = mind
                else:
                    model_input = img
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                        
                        val_loss, val_all_loss = self.__compute_loss(
                            self.loss_fn,
                            recon_patch,
                            orig_patch,
                            img,
                            multiple_mask,
                            dec_mask,
                            downsample=self.downsample,
                        )

                else:
                    dec_img, dec_mask, recon_patch, orig_patch = self.model(model_input)
                    val_loss, val_all_loss = self.__compute_loss(
                        self.loss_fn,
                        recon_patch,
                        orig_patch,
                        img,
                        multiple_mask,
                        dec_mask,
                        downsample=self.downsample,
                    )
                 
                val_loss_sum += val_loss.item()
                for l, loss_value in val_sub_loss_sum.items():
                    val_sub_loss_sum[l] = loss_value + val_all_loss[l]
                
        val_loss_mean = val_loss_sum / len(val_loader)
        val_sub_loss_mean = {
            l: vsls / len(val_loader) for l, vsls in val_sub_loss_sum.items()
        }
        print("Final validation Loss : {:.6f}".format(val_loss_mean))
        for l, vslm in val_sub_loss_mean.items():
            print("\t\t |- {} loss : {:.6f}".format(l, vslm))

        # save results
        results = pd.DataFrame(
            {
                "val": [
                    val_loss_mean,
                ],
            },
            index=["loss"],
        )
        results_sub_loss = pd.DataFrame(val_sub_loss_mean, index=["val"]).T
        results = pd.concat([results, results_sub_loss], axis=0)

        return results
    
    @staticmethod
    def __compute_loss(
        loss_fn, recon_patch, orig_patch, img, gt_mask, dec_mask, mode="train", downsample=1
    ):
        loss = 0.0
        all_loss = {}
        for l, lw in loss_fn.items():
            if l == "NCC":
                ncc_loss = lw[0](recon_patch, orig_patch) * lw[1]
                loss += ncc_loss
                all_loss["NCC"] = ncc_loss.item()
            elif l == "GNCC":
                gncc_loss = lw[0](recon_patch, orig_patch) * lw[1]
                loss += gncc_loss
                all_loss["GNCC"] = gncc_loss.item()
            elif l == "MSE":
                mse_loss = lw[0](recon_patch, orig_patch) * lw[1]
                loss += mse_loss
                all_loss["MSE"] = mse_loss.item()
            elif l == "SAD":
                sad_loss = lw[0](recon_patch, orig_patch) * lw[1]
                loss += sad_loss
                all_loss["SAD"] = sad_loss.item()
            elif l == "MINDSSC" or l == "MINDSSC_NCC":
                mindssc_loss = lw[0](recon_patch, orig_patch) * lw[1]
                loss += mindssc_loss
                all_loss[l] = mindssc_loss.item()
            elif l == "Seg_MSE":
                segmse_loss = lw[0](dec_mask, gt_mask) * lw[1]
                loss += segmse_loss
                all_loss[l] = segmse_loss.item()
        return loss, all_loss

    def load_prev(self):
        print("Continue training from checkpoint: {}".format(self.ckpt_path))
        if os.path.isfile(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path)
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scheduler:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            # self.writer = ckpt['writer']
            self.early_stopping = ckpt["early_stopping"]
        else:
            raise ValueError("No checkpoint found at '{}'".format(self.ckpt_path))
