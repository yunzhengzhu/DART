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
from model.mae_down import MAE_Finetune_Baseline
from model.vxm import Voxelmorph
from model.vitvnet import ViTVNet
from model.transform import SpatialTransform, DiffeomorphicTransform, ResizeTransform
from utils.loss_utils import smoothLoss, NCC, GNCC, Dice, MSE, SAD, TRE, MINDSSC, Seg_MSE 
from utils.train_utils import EarlyStopping
from utils.metric_utils import jacobian_determinant, compute_tre, compute_dice, compute_all_tre
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
        self.diff = args.diff if mode == "train" else args.eval_diff
        self.freeze = args.freeze
        self.pretrained = args.pretrained if mode == "train" else None
        self.rev_metric = args.rev_metric
        self.blur_factor = args.blur_factor if mode == "train" else args.eval_blur_factor
        self.es_criterion = args.es_criterion
        self.mind_feature = args.mind_feature
        self.masked_img = args.masked_img
        self.transform_type = args.transform_type
        self.use_augs = True if args.augs != None else False
        self.use_texture_mask = True if args.texture_mask_dir else False
        self.use_nodule_kpt = True if args.nodule_kp_dir else False
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
            elif l == "Smooth":
                self.loss_fn[l] = [smoothLoss(), lw]
            elif l == "Dice":
                self.loss_fn[l] = [Dice(), lw]
            elif l == "Seg_MSE":
                self.loss_fn[l] = [Seg_MSE(), lw]
            elif l == "MSE":
                self.loss_fn[l] = [MSE(), lw]
            elif l == "SAD":
                self.loss_fn[l] = [SAD(), lw]
            elif l == "TRE":
                self.loss_fn[l] = [TRE(), lw]
            elif l == "MINDSSC":
                self.loss_fn[l] = [MINDSSC(), lw]
            elif l == "MINDSSC_NCC":
                self.loss_fn[l] = [MINDSSC(loss_type="ncc"), lw]
            else:
                raise NotImplementedError
        print("...done")

    def __init_model(self):
        print(f"Initiate {self.model_type} model", end=" ")
        if self.mind_feature:
            in_channel = 12 * 2
        else:
            in_channel = 2
        
        if self.model_type == "Vxm":
            self.model = Voxelmorph(in_channels=in_channel)
            print(self.model)
        elif self.model_type == "ViT-V-Net":
            self.model = ViTVNet(in_channels=in_channel)
            print(self.model)
        elif self.model_type == "MAE_ViT_Baseline":
            self.model = MAE_Finetune_Baseline(
                in_channels=in_channel,
            )
            print(self.model)
        else:
            raise NotImplementedError
            
        print("...done")

        if self.pretrained:
            print("Load pretrained model", end=" ")
            if self.model_type == "MAE_ViT_Baseline":
                loaded_weights = torch.load(self.pretrained, map_location='cuda:0')["model"]
                model_dict = self.model.state_dict()
                mae_transformer_weights = {}
                mae_transformer_weights = {k: v for k, v in loaded_weights.items() if k in model_dict}
                assert mae_transformer_weights != {}, "Warning: Weights are not loaded successfully!"
                no_pt = [
                    'mae_transformer.mask_token',
                    'mae_transformer.conv3d_transpose.weight',
                    'mae_transformer.conv3d_transpose.bias',
                    'mae_transformer.conv3d_transpose1.weight',
                    'mae_transformer.conv3d_transpose1.bias',
                    'mae_transformer.conv3d_transpose_seg.weight'
                    'mae_transformer.conv3d_transpose_seg.bias'
                    'mae_transformer.conv3d_transpose_seg1.weight'
                    'mae_transformer.conv3d_transpose_seg1.bias'
                ]
                for name, param in mae_transformer_weights.items():
                    #if 'Transformer_encoder' in name:
                    #    model_dict[name] = param
                    #else:
                    #    continue
                    if name in no_pt:
                        continue
                    else:
                        model_dict[name] = param
                self.model.load_state_dict(model_dict)
                del model_dict, mae_transformer_weights, loaded_weights
            else: 
                self.model.load_state_dict(torch.load(self.pretrained, map_location='cuda:0')["model"])
            print("...done")
        
        if self.freeze:
            print(f"Freezing {self.freeze}")
            for name, param in self.model.named_parameters():
                if self.freeze in name:
                    param.requires_grad = False

        self.spatial_transform = SpatialTransform()
        if self.diff:
            self.diff_transform = DiffeomorphicTransform()
        
        if self.blur_factor:
            self.blur = ResizeTransform(factor=1/self.blur_factor)
            self.deblur = ResizeTransform(factor=self.blur_factor)
        else:
            self.blur = None
            self.deblur = None

        for param in self.spatial_transform.parameters():
            param.requires_grad = False
            param.volatile = True

        self.model.to(self.device)
        self.spatial_transform.to(self.device)
        if self.diff:
            self.diff_transform.to(self.device)

        if self.blur_factor:
            self.blur.to(self.device)
            self.deblur.to(self.device)

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
        self.mode = mode
    
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
            train_num_foldings, train_log_jac_det_std, train_tre, train_dice = [], [], [], []
            start_time = time.time()
            for batch_idx, batch_data in enumerate(train_loader):
                if self.use_augs:
                    if self.transform_type == "same":
                        fixed_img = batch_data['f_img'][tio.DATA]
                        fixed_mask = batch_data['f_mask'][tio.DATA]
                        fixed_multiple_mask = batch_data['f_mulmask'][tio.DATA]
                        fixed_kp = batch_data['f_kpt']
                        moving_img = batch_data['m_img'][tio.DATA]
                        moving_mask = batch_data['m_mask'][tio.DATA]
                        moving_multiple_mask = batch_data['m_mulmask'][tio.DATA]
                        moving_kp = batch_data['m_kpt']
                        mask_labels = batch_data['labels'][0][0]
                    elif self.transform_type == "diff":
                        fixed_img = batch_data[0]['f_img'][tio.DATA]
                        fixed_mask = batch_data[0]['f_mask'][tio.DATA]
                        fixed_multiple_mask = batch_data[0]['f_mulmask'][tio.DATA]
                        fixed_kp = batch_data[0]['f_kpt']
                        moving_img = batch_data[1]['m_img'][tio.DATA]
                        moving_mask = batch_data[1]['m_mask'][tio.DATA]
                        moving_multiple_mask = batch_data[1]['m_mulmask'][tio.DATA]
                        moving_kp = batch_data[1]['m_kpt']
                        mask_labels = batch_data[0]['labels'][0][0]
                else:
                    fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, mask_labels = batch_data
                fixed_img, moving_img = fixed_img.float().to(
                    self.device
                ), moving_img.float().to(self.device)
                fixed_kp, moving_kp = fixed_kp.float().to(
                    self.device
                ), moving_kp.float().to(self.device)
                fixed_mask, moving_mask = fixed_mask.float().to(
                    self.device
                ), moving_mask.float().to(self.device)
                fixed_multiple_mask, moving_multiple_mask = fixed_multiple_mask.float().to(
                    self.device
                ), moving_multiple_mask.float().to(self.device)
                
                # mind feature
                if self.masked_img:
                    fixed_img = fixed_img * fixed_mask
                    moving_img = moving_img * moving_mask

                if self.mind_feature:
                    fixed_mind = mindssc(fixed_img)
                    moving_mind = mindssc(moving_img)
                    model_input = (fixed_mind, moving_mind)
                else:
                    model_input = (fixed_img, moving_img)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        rf = self.model(model_input[0], model_input[1])
                        if self.blur:
                            rf = self.blur(rf)

                        if self.diff:
                            D_rf = self.diff_transform(rf)
                        else:
                            D_rf = rf

                        if self.deblur:
                            D_rf = self.deblur(D_rf)

                        warp_img = self.spatial_transform(
                            moving_img, D_rf.permute(0, 2, 3, 4, 1)
                        )
                        warp_mask = self.spatial_transform(
                            moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                        )

                        # compute metrics
                        (
                           batch_num_foldings,
                           batch_log_jac_det_std,
                           batch_tre,
                           batch_dice,
                        ) = self.__compute_metrics(
                           D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, warp_mask,
                           downsample=self.downsample, mode="train",
                        )
                        train_num_foldings.extend(batch_num_foldings)
                        train_log_jac_det_std.extend(batch_log_jac_det_std)
                        train_tre.extend(batch_tre)
                        train_dice.extend(batch_dice)
                        train_loss, train_all_loss = self.__compute_loss(
                            self.loss_fn,
                            fixed_img,
                            warp_img,
                            fixed_mask,
                            warp_mask,
                            fixed_kp,
                            moving_kp,
                            rf,
                            mode="train",
                            downsample=self.downsample,
                        )

                            

                    self.scaler.scale(train_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    rf = self.model(model_input[0], model_input[1])
                    if self.blur:
                        rf = self.blur(rf)

                    if self.diff:
                        D_rf = self.diff_transform(rf)
                    else:
                        D_rf = rf

                    if self.deblur:
                        D_rf = self.deblur(D_rf)

                    warp_img = self.spatial_transform(
                        moving_img, D_rf.permute(0, 2, 3, 4, 1)
                    )
                    warp_mask = self.spatial_transform(
                        moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest" 
                    )

                    # compute metrics
                    (
                       batch_num_foldings,
                       batch_log_jac_det_std,
                       batch_tre,
                       batch_dice,
                    ) = self.__compute_metrics(
                       D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, warp_mask,
                       downsample=self.downsample, mode="train",
                    )
                    train_num_foldings.extend(batch_num_foldings)
                    train_log_jac_det_std.extend(batch_log_jac_det_std)
                    train_tre.extend(batch_tre)
                    train_dice.extend(batch_dice)

                    train_loss, train_all_loss = self.__compute_loss(
                        self.loss_fn,
                        fixed_img,
                        warp_img,
                        fixed_mask,
                        warp_mask,
                        fixed_kp,
                        moving_kp,
                        rf,
                        mode="train",
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

            train_num_foldings_mean = np.mean(train_num_foldings)
            train_log_jac_det_std_mean = np.mean(train_log_jac_det_std)
            train_tre_mean = np.mean(train_tre)
            train_dice_mean = np.mean(train_dice)
            print("Epoch {} - Train Loss: {:.6f}; Dice: {:.6f}; TRE: {:.6f}; JacDet: {:.6f}".format(i, train_loss_mean, train_dice_mean, train_tre_mean, train_log_jac_det_std_mean))
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
                self.writer.add_scalar("train/Dice", train_dice_mean, i)
                self.writer.add_scalar("train/TRE", train_tre_mean, i)
                self.writer.add_scalar("train/num_foldings", train_num_foldings_mean, i)
                self.writer.add_scalar("train/log_jac_det_std", train_log_jac_det_std_mean, i)

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
        self.mode = "val"
        results = self.predict(val_loader)
        
        if self.writer:
            self.writer.close()

        return results

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader) -> bool:
        val_loss_sum = 0
        val_sub_loss_sum = {l: 0.0 for l in self.loss if l != "Smooth"}
        val_num_foldings, val_log_jac_det_std, val_tre, val_dice = [], [], [], []
        if self.rev_metric:
            rev_val_num_foldings, rev_val_log_jac_det_std, rev_val_tre, rev_val_dice = [], [], [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (
                fixed_img,
                moving_img,
                fixed_kp,
                moving_kp,
                fixed_mask,
                moving_mask,
                fixed_multiple_mask,
                moving_multiple_mask,
                mask_labels,
            ) in enumerate(val_loader):
                fixed_img, moving_img = fixed_img.float().to(
                    self.device
                ), moving_img.float().to(self.device)
                fixed_kp, moving_kp = fixed_kp.float().to(
                    self.device
                ), moving_kp.float().to(self.device)
                fixed_mask, moving_mask = fixed_mask.float().to(
                    self.device
                ), moving_mask.float().to(self.device)
                fixed_multiple_mask, moving_multiple_mask = fixed_multiple_mask.float().to(
                    self.device
                ), moving_multiple_mask.float().to(self.device)
                
                # mind feature
                if self.masked_img:
                    fixed_img = fixed_img * fixed_mask
                    moving_img = moving_img * moving_mask

                if self.mind_feature:
                    fixed_mind = mindssc(fixed_img)
                    moving_mind = mindssc(moving_img)
                    model_input = (fixed_mind, moving_mind)
                else:
                    model_input = (fixed_img, moving_img)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # pass data to model
                        rf = self.model(model_input[0], model_input[1])
                        if self.blur:
                            rf = self.blur(rf)

                        if self.diff:
                            D_rf = self.diff_transform(rf)
                        else:
                            D_rf = rf

                        if self.deblur:
                            D_rf = self.deblur(D_rf)

                        warp_img = self.spatial_transform(
                            moving_img, D_rf.permute(0, 2, 3, 4, 1)
                        )
                        warp_mask = self.spatial_transform(
                            moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                        )
                        val_loss, val_all_loss = self.__compute_loss(
                            self.loss_fn,
                            fixed_img,
                            warp_img,
                            fixed_mask,
                            warp_mask,
                            fixed_kp,
                            moving_kp,
                            rf,
                            mode="val",
                            downsample=self.downsample,
                        )
                else:
                    # pass data to model
                    rf = self.model(model_input[0], model_input[1])
                    
                    if self.blur:
                        rf = self.blur(rf)

                    if self.diff:
                        D_rf = self.diff_transform(rf)
                    else:
                        D_rf = rf

                    if self.deblur:
                        D_rf = self.deblur(D_rf)

                    warp_img = self.spatial_transform(
                        moving_img, D_rf.permute(0, 2, 3, 4, 1)
                    )
                    warp_mask = self.spatial_transform(
                        moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                    )
                    
                    val_loss, val_all_loss = self.__compute_loss(
                        self.loss_fn,
                        fixed_img,
                        warp_img,
                        fixed_mask,
                        warp_mask,
                        fixed_kp,
                        moving_kp,
                        rf,
                        mode="val",
                        downsample=self.downsample,
                    )

                
                # compute metrics
                (
                    batch_num_foldings,
                    batch_log_jac_det_std,
                    batch_tre,
                    batch_dice,
                ) = self.__compute_metrics(
                    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, warp_mask,
                    downsample=self.downsample, mode="val",
                )
                val_num_foldings.extend(batch_num_foldings)
                val_log_jac_det_std.extend(batch_log_jac_det_std)
                val_tre.extend(batch_tre)
                val_dice.extend(batch_dice)
                
                if self.rev_metric:
                    rrf = -rf
                    rrf = rrf.to(torch.float32)
                    if self.blur:
                        rrf = self.blur(rrf)

                    if self.diff:
                        D_rrf = self.diff_transform(rrf)
                    else:
                        D_rrf = rrf

                    if self.deblur:
                        D_rf = self.deblur(D_rf)

                    warp_reg_rev = self.spatial_transform(
                        fixed_img, D_rrf.permute(0, 2, 3, 4, 1)
                    )
                    warp_mask_rev = self.spatial_transform(
                        fixed_mask, D_rrf.permute(0, 2, 3, 4, 1), mod="nearest"
                    )
                    
                    # compute metrics from reverse direction
                    (
                        rev_batch_num_foldings,
                        rev_batch_log_jac_det_std,
                        rev_batch_tre,
                        rev_batch_dice,
                    ) = self.__compute_metrics(
                        D_rrf, moving_kp, fixed_kp, moving_mask, fixed_mask, warp_mask_rev,
                        downsample=self.downsample, mode="val",
                    )
                    rev_val_num_foldings.extend(rev_batch_num_foldings)
                    rev_val_log_jac_det_std.extend(rev_batch_log_jac_det_std)
                    rev_val_tre.extend(rev_batch_tre)
                    rev_val_dice.extend(rev_batch_dice)

                val_loss_sum += val_loss.item()
                for l, loss_value in val_sub_loss_sum.items():
                    val_sub_loss_sum[l] = loss_value + val_all_loss[l]

        val_loss_mean = val_loss_sum / len(val_loader)
        val_sub_loss_mean = {
            l: loss_value / len(val_loader)
            for l, loss_value in val_sub_loss_sum.items()
        }

        val_num_foldings_mean = np.mean(val_num_foldings)
        val_num_foldings_std = np.std(val_num_foldings)
        val_log_jac_det_std_mean = np.mean(val_log_jac_det_std)
        val_log_jac_det_std_std = np.std(val_log_jac_det_std)
        val_tre_mean = np.mean(val_tre)
        val_tre_std = np.std(val_tre)
        val_dice_mean = np.mean(val_dice)
        val_dice_std = np.std(val_dice)
        if self.rev_metric:
            rev_val_num_foldings_mean = np.mean(rev_val_num_foldings)
            rev_val_num_foldings_std = np.std(rev_val_num_foldings)
            rev_val_log_jac_det_std_mean = np.mean(rev_val_log_jac_det_std)
            rev_val_log_jac_det_std_std = np.std(rev_val_log_jac_det_std)
            rev_val_tre_mean = np.mean(rev_val_tre)
            rev_val_tre_std = np.std(rev_val_tre)
            rev_val_dice_mean = np.mean(rev_val_dice)
            rev_val_dice_std = np.std(rev_val_dice)
        print("Epoch {} - Validation Loss : {:.6f}".format(cur, val_loss_mean))
        for l, vslm in val_sub_loss_mean.items():
            print("\t\t |- {} loss: {:.6f}".format(l, vslm))
        print(
                "\t\t(fwd) Dice: {:.6f}({:.6f}); TRE: {:.6f}({:.6f}); NumFold: {:.6f}({:.6f}); LogJacDetStd: {:6f}({:.6f})".format(
                val_dice_mean,
                val_dice_std,
                val_tre_mean,
                val_tre_std,
                val_num_foldings_mean,
                val_num_foldings_std,
                val_log_jac_det_std_mean,
                val_log_jac_det_std_std,
            )
        )
        if self.rev_metric:
            print(
                "\t\t(rev) Dice: {:.6f}({:.6f}); TRE: {:.6f}({:.6f}); NumFold: {:.6f}({:.6f}); LogJacDetStd: {:.6f}({:.6f})".format(
                    rev_val_dice_mean,
                    rev_val_dice_std,
                    rev_val_tre_mean,
                    rev_val_tre_std,
                    rev_val_num_foldings_mean,
                    rev_val_num_foldings_std,
                    rev_val_log_jac_det_std_mean,
                    rev_val_log_jac_det_std_std,
                )
            )

        if self.writer:
            self.writer.add_scalar("val/Total_loss", val_loss_mean, cur)
            for l, vslm in val_sub_loss_mean.items():
                self.writer.add_scalar("val/{}_loss".format(l), vslm, cur)
            self.writer.add_scalar("val/Dice", val_dice_mean, cur)
            self.writer.add_scalar("val/TRE", val_tre_mean, cur)
            self.writer.add_scalar("val/num_foldings", val_num_foldings_mean, cur)
            self.writer.add_scalar("val/log_jac_det_std", val_log_jac_det_std_mean, cur)
            if self.rev_metric:
                self.writer.add_scalar("val/Dice_rev", rev_val_dice_mean, cur)
                self.writer.add_scalar("val/TRE_rev", rev_val_tre_mean, cur)
                self.writer.add_scalar("val/num_foldings_rev", rev_val_num_foldings_mean, cur)
                self.writer.add_scalar("val/log_jac_det_std_rev", rev_val_log_jac_det_std_mean, cur)


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
        
        if self.mode == "val" or self.mode == "test":
            #val_loss_sum = 0
            #val_sub_loss_sum = {l: 0.0 for l in self.loss if l != "Smooth"}
            val_num_foldings, val_log_jac_det_std, val_tre, val_dice = [], [], [], []
            if self.use_nodule_kpt:
                val_gt_nodule_tre = []
                val_nodule_all_tre = []
            if self.rev_metric: 
                rev_val_num_foldings, rev_val_log_jac_det_std, rev_val_tre, rev_val_dice = [], [], [], []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                if self.use_nodule_kpt:
                    fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, fixed_nodu_kp, moving_nodu_kp, mask_labels = batch_data
                else:
                    fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask, fixed_multiple_mask, moving_multiple_mask, mask_labels = batch_data
                fixed_img, moving_img = fixed_img.float().to(
                    self.device
                ), moving_img.float().to(self.device)
                fixed_kp, moving_kp = fixed_kp.float().to(
                    self.device
                ), moving_kp.float().to(self.device)
                fixed_mask, moving_mask = fixed_mask.float().to(
                    self.device
                ), moving_mask.float().to(self.device)
                fixed_multiple_mask, moving_multiple_mask = fixed_multiple_mask.float().to(
                    self.device
                ), moving_multiple_mask.float().to(self.device)
                if self.use_nodule_kpt:
                    fixed_nodu_kp, moving_nodu_kp = fixed_nodu_kp.float().to(
                        self.device
                    ), moving_nodu_kp.float().to(self.device)
                
                # mind feature
                if self.masked_img:
                    fixed_img = fixed_img * fixed_mask
                    moving_img = moving_img * moving_mask

                if self.mind_feature:
                    fixed_mind = mindssc(fixed_img)
                    moving_mind = mindssc(moving_img)
                    model_input = (fixed_mind, moving_mind)
                else:
                    model_input = (fixed_img, moving_img)                

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # pass data to model
                        rf = self.model(model_input[0], model_input[1])
                        if self.blur:
                            rf = self.blur(rf)

                        if self.diff:
                            D_rf = self.diff_transform(rf)
                        else:
                            D_rf = rf

                        if self.deblur:
                            D_rf = self.deblur(D_rf)
            

                        warp_img = self.spatial_transform(
                            moving_img, D_rf.permute(0, 2, 3, 4, 1)
                        )
                        warp_mask = self.spatial_transform(
                            moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                        )
                else:
                    # pass data to model
                    rf = self.model(model_input[0], model_input[1])
                    if self.blur:
                        rf = self.blur(rf)

                    if self.diff:
                        D_rf = self.diff_transform(rf)
                    else:
                        D_rf = rf

                    if self.deblur:
                        D_rf = self.deblur(D_rf)
                        
                    warp_img = self.spatial_transform(
                        moving_img, D_rf.permute(0, 2, 3, 4, 1)
                    )
                    warp_mask = self.spatial_transform(
                        moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                    )

                # compute metrics
                if self.mode == "val" or self.mode == "test":
                    if self.use_nodule_kpt:
                        # compute the gt kpt tre
                        batch_gt_nodule_tre = []
                        for subject_idx in range(fixed_nodu_kp.shape[1]):
                            nodule_tre = compute_tre(
                                fix_lms=fixed_nodu_kp[subject_idx].clone().detach().cpu().numpy(),
                                mov_lms=moving_nodu_kp[subject_idx].clone().detach().cpu().numpy(),
                                disp=torch.zeros_like(D_rf)[subject_idx]
                                .clone()
                                .detach()
                                .cpu()
                                .numpy(),
                                spacing_fix=(1.5, 1.5, 1.5),
                                spacing_mov=(1.5, 1.5, 1.5),
                            )  # spacing is 1.5 for NLST
                            batch_gt_nodule_tre.append(nodule_tre)

                            # save cases tre
                            # recover to the shape of original image
                            tmp_D_rf = D_rf.clone()
                            H, W, D = tmp_D_rf.shape[2:]
                            if self.downsample != 1:
                                tmp_D_rf = F.interpolate(tmp_D_rf, scale_factor=self.downsample, align_corners=True, mode="trilinear")

                            # denormalize the disp (to the scale of training images)
                            tmp_D_rf = ((tmp_D_rf.permute(0, 2, 3, 4, 1)) * (torch.tensor([D, W, H]).cuda()-1)).flip(-1).float()
                            tre_all = compute_all_tre(
                                fix_lms=fixed_nodu_kp[subject_idx].clone().detach().cpu().numpy(),
                                mov_lms=moving_nodu_kp[subject_idx].clone().detach().cpu().numpy(),
                                disp=tmp_D_rf[subject_idx]
                                .clone()
                                .detach()
                                .cpu()
                                .numpy(),
                                spacing_fix=(1.5, 1.5, 1.5),
                                spacing_mov=(1.5, 1.5, 1.5),
                            )  # spacing is 1.5 for NLST
                            print(tre_all.tolist()[0])
                            val_nodule_all_tre.append(tre_all.tolist()[0])
                    
                    # fwd metrics
                    (
                        batch_num_foldings,
                        batch_log_jac_det_std,
                        batch_tre,
                        batch_dice,
                    ) = self.__compute_metrics(
                        D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, warp_mask,
                        downsample=self.downsample, mode="val",
                    )
                    val_num_foldings.extend(batch_num_foldings)
                    val_log_jac_det_std.extend(batch_log_jac_det_std)
                    if self.use_nodule_kpt:
                        val_gt_nodule_tre.extend(batch_gt_nodule_tre)
                    val_tre.extend(batch_tre)
                    val_dice.extend(batch_dice)

                    
                    # rev metrics
                    if self.rev_metric:
                        rrf = -rf
                        rrf = rrf.to(torch.float32)
                        if self.blur:
                            rrf = self.blur(rrf)

                        if self.diff:
                            D_rrf = self.diff_transform(rrf)
                        else:
                            D_rrf = rrf

                        if self.deblur:
                            D_rrf = self.deblur(D_rrf)

                        warp_img_rev = self.spatial_transform(
                            fixed_img, D_rrf.permute(0, 2, 3, 4, 1)
                        )
                        warp_mask_rev = self.spatial_transform(
                            fixed_mask, D_rrf.permute(0, 2, 3, 4, 1), mod="nearest"
                        )
                        
                        # compute metrics from reverse direction
                        (
                            rev_batch_num_foldings,
                            rev_batch_log_jac_det_std,
                            rev_batch_tre,
                            rev_batch_dice,
                        ) = self.__compute_metrics(
                            D_rrf, moving_kp, fixed_kp, moving_mask, fixed_mask, warp_mask_rev,
                            downsample=self.downsample, mode="val",
                        )
                        rev_val_num_foldings.extend(rev_batch_num_foldings)
                        rev_val_log_jac_det_std.extend(rev_batch_log_jac_det_std)
                        rev_val_tre.extend(rev_batch_tre)
                        rev_val_dice.extend(rev_batch_dice)

                # masking disp
                #D_rf = D_rf * fixed_mask
                
                # save displacement field 
                if self.save_df or self.save_warped:
                    for iidx, subject in enumerate(
                        val_loader.dataset.subjects[
                            batch_idx
                            * val_loader.batch_size : (batch_idx + 1)
                            * val_loader.batch_size
                        ]
                    ):
                        H = val_loader.dataset.H // self.downsample
                        W = val_loader.dataset.W // self.downsample
                        D = val_loader.dataset.D // self.downsample

                        subject_id = subject["moving"].split("/")[-1].split("_")[1]
                        if self.save_df:
                            save_df_fn = f"displacement_field_{self.mode}"
                            os.makedirs(
                                os.path.join(self.exp_dir, save_df_fn), exist_ok=True
                            )
                            if self.downsample > 1:
                                D_rf = F.interpolate(D_rf,scale_factor=self.downsample,align_corners=True,mode='trilinear')
                            D_rf=((D_rf.permute(0,2,3,4,1))*(torch.tensor([D,W,H]).cuda()-1)).flip(-1).float().squeeze().cpu()
                            nib.save(nib.Nifti1Image(D_rf.numpy(), np.eye(4)), 
                                     os.path.join(self.exp_dir,save_df_fn, 
                                                  f'disp_{str(subject_id).zfill(4)}_{str(subject_id).zfill(4)}.nii.gz'))
                        
                        if self.save_warped:
                            save_warped_fn = f"warped_results_{self.mode}"
                            os.makedirs(
                                os.path.join(self.exp_dir, save_warped_fn), exist_ok=True
                            )
                            if self.downsample > 1:
                                warp_img = F.interpolate(warp_img,scale_factor=self.downsample,align_corners=True,mode='trilinear')
                            warp_img = warp_img.squeeze().cpu()
                            # denormalize warped
                            warp_img = (warp_img * (500 - (-1000)) + (-1000)).to(torch.int16)
                            aff = np.eye(4) * 1.5
                            aff[3, 3] = 1.0
                            nib.save(nib.Nifti1Image(warp_img.numpy(), aff),
                                    os.path.join(self.exp_dir,save_warped_fn,
                                                 f'warped_{str(subject_id).zfill(4)}_{str(subject_id).zfill(4)}.nii.gz'))


        if self.mode == "val" or self.mode == "test":
            val_num_foldings_mean = np.mean(val_num_foldings)
            val_num_foldings_std = np.std(val_num_foldings)
            val_log_jac_det_std_mean = np.mean(val_log_jac_det_std)
            val_log_jac_det_std_std = np.std(val_log_jac_det_std)
            val_tre_mean = np.nanmean(val_tre)
            val_tre_std = np.nanstd(val_tre)
            val_dice_mean = np.mean(val_dice)
            val_dice_std = np.std(val_dice)
            if self.use_nodule_kpt:
                val_gt_nodule_tre_mean = np.nanmean(val_gt_nodule_tre)
                val_gt_nodule_tre_std = np.nanstd(val_gt_nodule_tre)
                val_reg_nodule_tre_mean = np.nanmean(np.array([
                    np.nanmean(np.array(nodalltre)) for nodalltre in val_nodule_all_tre if nodalltre
                ]))
                val_reg_nodule_tre_std = np.nanstd(np.array([
                    np.nanmean(np.array(nodalltre)) for nodalltre in val_nodule_all_tre if nodalltre
                ]))
            if self.rev_metric:
                rev_val_num_foldings_mean = np.mean(rev_val_num_foldings)
                rev_val_num_foldings_std = np.std(rev_val_num_foldings)
                rev_val_log_jac_det_std_mean = np.mean(rev_val_log_jac_det_std)
                rev_val_log_jac_det_std_std = np.std(rev_val_log_jac_det_std)
                rev_val_tre_mean = np.nanmean(rev_val_tre)
                rev_val_tre_std = np.nanstd(rev_val_tre)
                rev_val_dice_mean = np.mean(rev_val_dice)
                rev_val_dice_std = np.std(rev_val_dice)

            print("Final validation Result:")
            print(
                "\t\t(fwd) Dice: {:.6f}({:.6f}); TRE: {:.6f}({:.6f}); NumFold: {:.6f}({:.6f}); LogJacDetStd: {:.6f}({:.6f})".format(
                    val_dice_mean,
                    val_dice_std,
                    val_tre_mean,
                    val_tre_std,
                    val_num_foldings_mean,
                    val_num_foldings_std,
                    val_log_jac_det_std_mean,
                    val_log_jac_det_std_std,
                )
            )
            if self.use_nodule_kpt:
                print(
                    "\t\t GT Nodule TRE: {:.6f}({:.6f}) Reg Nodule TRE: {:.6f}({:.6f})".format(
                        val_gt_nodule_tre_mean, 
                        val_gt_nodule_tre_std, 
                        val_reg_nodule_tre_mean, 
                        val_reg_nodule_tre_std
                    )
                )
            if self.rev_metric:
                print(
                    "\t\t(rev) Dice: {:.6f}({:.6f}); TRE: {:.6f}({:.6f}); NumFold: {:.6f}({:.6f}); LogJacDetStd: {:.6f}({:.6f})".format(
                        val_dice_mean,
                        val_dice_std,
                        val_tre_mean,
                        val_tre_std,
                        val_num_foldings_mean,
                        val_num_foldings_std,
                        val_log_jac_det_std_mean,
                        val_log_jac_det_std_std,
                    )
                )

            # save results
            results = pd.DataFrame(
                {
                    f"{self.mode}_mean": [
                        val_dice_mean,
                        val_tre_mean,
                        val_num_foldings_mean,
                        val_log_jac_det_std_mean,
                    ],
                    f"{self.mode}_std": [
                        val_dice_std,
                        val_tre_std,
                        val_num_foldings_std,
                        val_log_jac_det_std_std,
                    ],
                },
                index=["dice", "tre", "num_fold", "log_jac_det_std"],
            )
            if self.use_nodule_kpt:
                gt_nodule_results = pd.DataFrame(
                    {
                        f"{self.mode}_mean": [
                            val_gt_nodule_tre_mean,
                        ],
                        f"{self.mode}_std": [
                            val_gt_nodule_tre_std,
                        ],
                    },
                    index=["gt_nodule_tre"],
                )
                results = pd.concat([results, gt_nodule_results], axis=0)
                reg_nodule_results = pd.DataFrame(
                    {
                        f"{self.mode}_mean": [
                            val_reg_nodule_tre_mean,
                        ],
                        f"{self.mode}_std": [
                            val_reg_nodule_tre_std,
                        ],
                    },
                    index=["reg_nodule_tre"],
                )
                results = pd.concat([results, reg_nodule_results], axis=0)
                for fid in range(len(val_nodule_all_tre)):
                    all_nodule_tres = pd.DataFrame(
                        {
                            f"{self.mode}_case_tre": [val_nodule_all_tre[fid]],
                        },
                        index=[f"case{fid}"],
                    )
                    results = pd.concat([results, all_nodule_tres], axis=0)

            if self.rev_metric:
                rev_results = pd.DataFrame(
                    {
                        f"{self.mode}_mean": [
                            rev_val_dice_mean,
                            rev_val_tre_mean,
                            rev_val_num_foldings_mean,
                            rev_val_log_jac_det_std_mean,
                        ],
                        f"{self.mode}_std": [
                            rev_val_dice_std,
                            rev_val_tre_std,
                            rev_val_num_foldings_std,
                            rev_val_log_jac_det_std_std,
                        ],

                    },
                    index=["rev_dice", "rev_tre", "rev_num_fold", "rev_log_jac_det_std"],
                )
                results = pd.concat([results, rev_results], axis=0)

            return results
    
    @staticmethod
    def __compute_metrics(
        D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, warp_mask,
        downsample=1, use_mask='fixed', mode='train',
    ):
        batch_num_foldings, batch_log_jac_det_std, batch_tre, batch_dice = (
            [],
            [],
            [],
            [],
        )
        H, W, D = D_rf.shape[2:]
        # recover to the shape of original image
        if downsample != 1:
            D_rf = F.interpolate(D_rf, scale_factor=downsample, align_corners=True, mode="trilinear")
            fixed_mask = F.interpolate(fixed_mask, scale_factor=downsample, align_corners=True, mode="trilinear")
            moving_mask = F.interpolate(moving_mask, scale_factor=downsample, align_corners=True, mode="trilinear")
            warp_mask = F.interpolate(warp_mask, scale_factor=downsample, align_corners=True, mode="trilinear")

        # denormalize the disp (to the scale of training images)
        D_rf = ((D_rf.permute(0, 2, 3, 4, 1)) * (torch.tensor([D, W, H]).cuda()-1)).flip(-1).float()
        for subject_idx in range(len(D_rf)):
            if mode == 'val':
                # jacobian determinant
                num_foldings, log_jac_det = jacobian_determinant(
                    D_rf[subject_idx : subject_idx + 1].permute(0, 4, 1, 2, 3)
                    .clone()
                    .detach()
                    .cpu()
                    .numpy()
                )
                
                if use_mask != None:
                    mask = fixed_mask if use_mask == 'fixed' else moving_mask 
                    log_jac_det_std = np.ma.MaskedArray(
                        log_jac_det, 1-mask.squeeze().clone().detach().cpu().numpy()[2:-2, 2:-2, 2:-2]
                    ).std()
                else:
                    log_jac_det_std = log_jac_det.std()
            else:
                num_foldings = 0
                log_jac_det_std = 0
                  
            # TRE keypoints
            tre = compute_tre(
                fix_lms=fixed_kp[subject_idx].clone().detach().cpu().numpy(),
                mov_lms=moving_kp[subject_idx].clone().detach().cpu().numpy(),
                disp=D_rf[subject_idx]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                spacing_fix=(1.5, 1.5, 1.5),
                spacing_mov=(1.5, 1.5, 1.5),
            )  # spacing is 1.5 for NLST
          
            # Dice masks
            mean_dice, dice = compute_dice(
                fixed=fixed_mask[subject_idx].clone().detach().cpu().numpy(),
                moving=moving_mask[subject_idx].clone().detach().cpu().numpy(),
                warped=warp_mask[subject_idx]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                labels=[1],
            )  # labels is 1 for NLST
            batch_num_foldings.append(num_foldings)
            batch_log_jac_det_std.append(log_jac_det_std)
            batch_tre.append(tre)
            batch_dice.append(mean_dice)
        return batch_num_foldings, batch_log_jac_det_std, batch_tre, batch_dice
                        
    @staticmethod
    def __compute_loss(
        loss_fn, gt_img, warp_img, gt_mask, warp_mask, fixed_kp, moving_kp, rf, mode="train", downsample=1
    ):
        loss = 0.0
        all_loss = {}
        for l, lw in loss_fn.items():
            if l == "NCC":
                ncc_loss = lw[0](gt_img, warp_img) * lw[1]
                loss += ncc_loss
                all_loss["NCC"] = ncc_loss.item()
            elif l == "GNCC":
                gncc_loss = lw[0](gt_img, warp_img) * lw[1]
                loss += gncc_loss
                all_loss["GNCC"] = gncc_loss.item()
            elif l == "Smooth":
                if mode == "train":
                    smooth_loss = lw[0](rf) * lw[1]
                    loss += smooth_loss
                    all_loss["Smooth"] = smooth_loss.item()
                else:
                    pass
            elif l == "Dice":
                dice_loss = lw[0](gt_mask, warp_mask) * lw[1]
                loss += dice_loss
                all_loss["Dice"] = dice_loss.item()
            elif l == "Seg_MSE":
                assert gt_mask.shape[1] == warp_mask.shape[1]
                segmse_loss = []
                for nmask in gt_mask.shape[1]:
                    this_segmse_loss = lw[0](gt_mask[:, nmask:nmask+1, :, :, :], warp_mask[:, nmask:nmask+1, :, :, :]) * lw[1]
                    segmse_loss.append(this_segmse_loss.item())
                    loss += segmse_loss
                all_loss["Seg_MSE"] = torch.mean(segmse_loss)
            elif l == "MSE":
                mse_loss = lw[0](gt_img, warp_img) * lw[1]
                loss += mse_loss
                all_loss["MSE"] = mse_loss.item()
            elif l == "SAD":
                sad_loss = lw[0](gt_img, warp_img) * lw[1]
                loss += sad_loss
                all_loss["SAD"] = sad_loss.item()
            elif l == "TRE":
                tre_loss = lw[0](
                    fix_lms=fixed_kp,
                    mov_lms=moving_kp,
                    disp=rf,
                    spacing_fix=1.5,
                    spacing_mov=1.5,
                    downsample=downsample
                ) * lw[1]
                loss += tre_loss
                all_loss["TRE"] = tre_loss.item()
            elif l == "MINDSSC" or l == "MINDSSC_NCC":
                mindssc_loss = lw[0](gt_img, warp_img) * lw[1]
                loss += mindssc_loss
                all_loss[l] = mindssc_loss.item()

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
