import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import time
from argparse import ArgumentParser
from model.lkunet import LKUNet
from model.transform import SpatialTransform, DiffeomorphicTransform
from utils.loss_utils import smoothLoss, NCC, Dice, MSE, SAD
from utils.train_utils import EarlyStopping
from utils.metric_utils import jacobian_determinant, compute_tre, compute_dice
from tensorboardX import SummaryWriter


class baseTrainer:
    def __init__(self, args: ArgumentParser) -> None:
        self.args = args
        self.opt = args.opt
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.loss = args.loss
        self.model_type = args.model_type
        self.es = args.es
        self.es_warmup = args.es_warmup
        self.es_tolerence = args.es_tolerence
        self.start_channel = args.start_channel
        self.exp_dir = args.exp_dir
        self.log = args.log

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__init_model()
        self.__init_sim_loss()
        self.__init_smooth_loss()
        self.__init_optimizer()
        self.__init_logger()
        self.__init_es()

    def __init_optimizer(self):
        print(f"Initiate {self.opt} optimizer", end=" ")

        if self.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        print("...done")

    def __init_sim_loss(self):
        print(f"Initiate {self.loss} similarity loss", end=" ")
        if self.loss == "NCC":
            self.sim_loss = NCC()
        elif self.loss == "Dice":
            self.sim_loss = Dice()
        elif self.loss == "SAD":
            self.sim_loss = SAD()
        elif self.loss == "MSE":
            self.sim_loss = MSE()
        else:
            raise NotImplementedError

        print("...done")

    def __init_smooth_loss(self):
        print(f"Initiate smoothness loss", end=" ")
        self.smooth_loss = smoothLoss()
        print("...done")

    def __init_model(self):
        print(f"Initiate {self.model_type} model", end=" ")
        if self.model_type == "LKU-Net":
            self.model = LKUNet(
                in_channel=2, n_classes=3, start_channel=self.start_channel
            )
            print(self.model)
        else:
            raise NotImplementedError
        print("...done")

        self.spatial_transform = SpatialTransform()
        self.diff_transform = DiffeomorphicTransform()

        for param in self.spatial_transform.parameters():
            param.requires_grad = False
            param.volatile = True

        self.model.to(self.device)
        self.spatial_transform.to(self.device)
        self.diff_transform.to(self.device)

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
                "Initiate early stopping with warmup: {} and tolerence: {}".format(
                    self.es_warmup, self.es_tolerence
                ),
                end=" ",
            )
            self.early_stopping = EarlyStopping(
                warmup=self.es_warmup, tolerence=self.es_tolerence, verbose=True
            )
            print("...done")
        else:
            self.early_stopping = None


class Trainer(baseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.epochs = args.epochs
        self.smooth_w = args.smooth_w
        self.ckpt_path = os.path.join(args.exp_dir, "checkpoint.pt")

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        self.model.train()
        # loop through epochs
        for i in range(self.epochs):
            # training
            train_loss_sum = 0
            train_jac_det, train_tre, train_dice = [], [], []
            for batch_idx, (
                fixed_img,
                moving_img,
                fixed_kp,
                moving_kp,
                fixed_mask,
                moving_mask,
            ) in enumerate(train_loader):
                fixed_img, moving_img = fixed_img.float().to(
                    self.device
                ), moving_img.float().to(self.device)
                fixed_kp, moving_kp = fixed_kp.float().to(
                    self.device
                ), moving_kp.float().to(self.device)
                fixed_mask, moving_mask = fixed_mask.float().to(
                    self.device
                ), moving_mask.float().to(self.device)

                rf = self.model(moving_img, fixed_img)
                D_rf = self.diff_transform(rf)
                moving_reg = self.spatial_transform(
                    moving_img, D_rf.permute(0, 2, 3, 4, 1)
                )
                moving_mask_reg = self.spatial_transform(
                    moving_mask, D_rf.permute(0, 2, 3, 4, 1)
                )

                # compute metrics
                # batch_jac_det, batch_tre, batch_dice = self.__compute_metrics(D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg)
                # train_jac_det.extend(batch_jac_det)
                # train_tre.extend(batch_tre)
                # train_dice.extend(batch_dice)

                train_loss = self.sim_loss(
                    fixed_img, moving_reg
                ) + self.smooth_w * self.smooth_loss(rf)
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss_sum += train_loss.item()

                # training progess - per 10 batches
                if batch_idx % 20 == 0:
                    print(f"----batch {batch_idx}----")

            train_loss_mean = train_loss_sum / len(train_loader)
            # train_jac_det_mean = np.mean(train_jac_det)
            # train_tre_mean = np.mean(train_tre)
            # train_dice_mean = np.mean(train_dice)
            # print("Epoch {} - Train Loss: {:.6f}; Dice: {:.6f}; TRE: {:.6f}; JacDet: {:.6f}".format(i, train_loss_mean, train_jac_det_mean, train_tre_mean, train_dice_mean))
            print("Epoch {} - Train Loss: {:.6f}".format(i, train_loss_mean))

            # log training
            if self.writer:
                self.writer.add_scalar("train/loss", train_loss_mean, i)

            # validation
            earlystop = self.__eval(i, val_loader)
            if earlystop:
                break

        # if no earlystopping, we save the weights of the last epoch
        if not self.es:
            torch.save(self.model.state_dict(), self.ckpt_path)

        results = self.predict(val_loader)

        if self.writer:
            self.writer.close()

        return results

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader):
        val_loss_sum = 0
        val_jac_det, val_tre, val_dice = [], [], []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (
                fixed_img,
                moving_img,
                fixed_kp,
                moving_kp,
                fixed_mask,
                moving_mask,
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

                # pass data to model
                rf = self.model(moving_img, fixed_img)
                D_rf = self.diff_transform(rf)
                moving_reg = self.spatial_transform(
                    moving_img, D_rf.permute(0, 2, 3, 4, 1)
                )
                moving_mask_reg = self.spatial_transform(
                    moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                )

                # compute metrics
                batch_jac_det, batch_tre, batch_dice = self.__compute_metrics(
                    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
                )
                val_jac_det.extend(batch_jac_det)
                val_tre.extend(batch_tre)
                val_dice.extend(batch_dice)

                val_loss = self.sim_loss(
                    fixed_img, moving_reg
                ) + self.smooth_w * self.smooth_loss(rf)
                val_loss_sum += val_loss.item()
        val_loss_mean = val_loss_sum / len(val_loader)
        val_jac_det_mean = np.mean(val_jac_det)
        val_tre_mean = np.mean(val_tre)
        val_dice_mean = np.mean(val_dice)
        print(
            "Validation Loss: {:.6f}; Dice: {:.6f}; TRE: {:.6f}; JacDet: {:.6f}".format(
                val_loss_mean, val_dice_mean, val_tre_mean, val_jac_det_mean
            )
        )

        if self.writer:
            self.writer.add_scalar("val/loss", val_loss_mean, cur)
            self.writer.add_scalar("val/dice", val_dice_mean, cur)
            self.writer.add_scalar("val/tre", val_tre_mean, cur)
            self.writer.add_scalar("val/jac_det", val_jac_det_mean, cur)

        # early stopping
        if self.es:
            earlystop = self.early_stopping(
                epoch=cur,
                val_loss=val_loss_mean,
                model=self.model,
                ckpt_path=self.ckpt_path,
            )
            return earlystop
        else:
            return False

    def predict(self, val_loader):
        # load final model
        print("----Load Model Checkpoint----")
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

        val_loss_sum = 0
        val_jac_det, val_tre, val_dice = [], [], []
        with torch.no_grad():
            for batch_idx, (
                fixed_img,
                moving_img,
                fixed_kp,
                moving_kp,
                fixed_mask,
                moving_mask,
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

                # pass data to model
                rf = self.model(moving_img, fixed_img)
                D_rf = self.diff_transform(rf)
                moving_reg = self.spatial_transform(
                    moving_img, D_rf.permute(0, 2, 3, 4, 1)
                )
                moving_mask_reg = self.spatial_transform(
                    moving_mask, D_rf.permute(0, 2, 3, 4, 1), mod="nearest"
                )

                # compute metrics
                batch_jac_det, batch_tre, batch_dice = self.__compute_metrics(
                    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
                )
                val_jac_det.extend(batch_jac_det)
                val_tre.extend(batch_tre)
                val_dice.extend(batch_dice)

                val_loss = self.sim_loss(
                    fixed_img, moving_reg
                ) + self.smooth_w * self.smooth_loss(rf)
                val_loss_sum += val_loss.item()
        val_loss_mean = val_loss_sum / len(val_loader)
        val_jac_det_mean = np.mean(val_jac_det)
        val_tre_mean = np.mean(val_tre)
        val_dice_mean = np.mean(val_dice)

        print(
            "Final Loss: {:.6f}; Dice: {:.6f}; TRE: {:.6f}; JacDet: {:.6f}".format(
                val_loss_mean, val_dice_mean, val_tre_mean, val_jac_det_mean
            )
        )

        # save results
        results = pd.DataFrame(
            {
                "val": [val_loss_mean, val_dice_mean, val_tre_mean, val_jac_det_mean],
            },
            index=["loss", "dice", "tre", "jac_det"],
        )

        return results

    @staticmethod
    def __compute_metrics(
        D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
    ):
        batch_jac_det, batch_tre, batch_dice = [], [], []

        for subject_idx in range(len(D_rf)):
            # jacobian determinant
            jac_det = jacobian_determinant(
                D_rf[subject_idx : subject_idx + 1].clone().detach().cpu().numpy()
            )

            # TRE keypoints
            tre = compute_tre(
                fix_lms=fixed_kp[subject_idx].clone().detach().cpu().numpy(),
                mov_lms=moving_kp[subject_idx].clone().detach().cpu().numpy(),
                disp=D_rf[subject_idx].clone().detach().cpu().numpy(),
                spacing_fix=1.5,
                spacing_mov=1.5,
            )  # spacing is 1.5 for NLST

            # Dice masks
            mean_dice, dice = compute_dice(
                fixed=fixed_mask[subject_idx].clone().detach().cpu().numpy(),
                moving=moving_mask[subject_idx].clone().detach().cpu().numpy(),
                moving_warped=moving_mask_reg[subject_idx]
                .clone()
                .detach()
                .cpu()
                .numpy(),
                labels=[1],
            )  # labels is 1 for NLST
            batch_jac_det.append(jac_det)
            batch_tre.append(tre)
            batch_dice.append(mean_dice)
        return batch_jac_det, batch_tre, batch_dice
