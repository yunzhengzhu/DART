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
        self.loss_weight = args.loss_weight
        self.model_type = args.model_type
        self.es = args.es
        self.es_warmup = args.es_warmup
        self.es_patience = args.es_patience
        self.start_channel = args.start_channel
        self.exp_dir = args.exp_dir
        self.log = args.log
        self.print_every = args.print_every

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__init_model()
        self.__init_loss()
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

    def __init_loss(self):
        print(f"Initiate {self.loss} loss with weight {self.loss_weight}", end=" ")
        self.loss_weight = self.loss_weight / np.sum(self.loss_weight)
        assert len(self.loss) == len(
            self.loss_weight
        ), "Loss and loss weight must have the same length"
        self.loss_fn = {}
        for l, lw in zip(self.loss, self.loss_weight):
            if l == "NCC":
                self.loss_fn[l] = [NCC(), lw]
            elif l == "Smooth":
                self.loss_fn[l] = [smoothLoss(), lw]
            elif l == "Dice":
                self.loss_fn[l] = [Dice(), lw]
            elif l == "MSE":
                self.loss_fn[l] = [MSE(), lw]
            elif l == "SAD":
                self.loss_fn[l] = [SAD(), lw]
            else:
                raise NotImplementedError
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
                "Initiate early stopping with warmup: {} and patience: {}".format(
                    self.es_warmup, self.es_patience
                ),
                end=" ",
            )
            self.early_stopping = EarlyStopping(
                warmup=self.es_warmup, patience =self.es_patience, verbose=True
            )
            print("...done")
        else:
            self.early_stopping = None


class Trainer(baseTrainer):
    def __init__(self, args) -> None:
        super(Trainer, self).__init__(args)
        self.epochs = args.epochs
        self.ckpt_path = os.path.join(args.exp_dir, "checkpoint.pt")
        self.save_df = args.save_df

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> pd.DataFrame:
        self.model.train()
        # loop through epochs
        for i in range(self.epochs):
            print(f"------------Epoch {i+1}/{self.epochs}------------")
            # training
            train_loss_sum = 0
            train_sub_loss_sum = {l: 0.0 for l in self.loss}
            # train_jac_det, train_tre, train_dice = [], [], []
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
                #(
                #    batch_jac_det,
                #    batch_jac_det_std,
                #    batch_tre,
                #    batch_dice,
                #) = self.__compute_metrics(
                #    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
                #)
                # train_jac_det.extend(batch_jac_det)
                # train_tre.extend(batch_tre)
                # train_dice.extend(batch_dice)

                train_loss, train_all_loss = self.__compute_loss(
                    self.loss_fn, fixed_img, moving_reg, fixed_mask, moving_mask_reg, rf
                )

                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss_sum += train_loss.item()
                for l , loss_value in train_sub_loss_sum.items():
                    train_sub_loss_sum[l] = (
                        loss_value + train_all_loss[l]
                    )

                # training progess batch and loss
                if batch_idx % self.print_every == 0:
                    print(
                        "Batch {} - Train Loss: {:.6f}".format(
                            batch_idx, train_loss.item()
                        )
                    )
                    for l in self.loss:
                        print('\t\t |- {} loss: {:.6f}'.format(l, train_all_loss[l]))

            train_loss_mean = train_loss_sum / len(train_loader)
            train_sub_loss_mean = {l: loss_value/ len(train_loader) for l, loss_value in train_sub_loss_sum.items()}

            # train_jac_det_mean = np.mean(train_jac_det)
            # train_tre_mean = np.mean(train_tre)
            # train_dice_mean = np.mean(train_dice)
            # print("Epoch {} - Train Loss: {:.6f}; Dice: {:.6f}; TRE: {:.6f}; JacDet: {:.6f}".format(i, train_loss_mean, train_jac_det_mean, train_tre_mean, train_dice_mean))
            print("Epoch {} - Train Loss: {:.6f}".format(i, train_loss_mean))
            for l , tlm in train_sub_loss_mean.items():
                print('\t\t |- {} loss: {:.6f}'.format(l, tlm))

            # log training
            if self.writer:
                self.writer.add_scalar("train/Total_loss", train_loss_mean, i)
                for l , tlm in train_sub_loss_mean.items():
                    self.writer.add_scalar("train/{}_loss".format(l), tlm, i)

            # validation
            earlystop = self.__eval(i, val_loader)
            if earlystop:
                break

        # if no earlystopping, we save the weights of the last epoch
        if not self.es or i < self.warmup:
            torch.save(self.model.state_dict(), self.ckpt_path)

        print('Finished Training...')
        results = self.predict(val_loader)

        if self.writer:
            self.writer.close()

        return results

    def __eval(self, cur, val_loader: torch.utils.data.DataLoader) -> bool:
        val_loss_sum = 0
        val_sub_loss_sum = {l: 0.0 for l in self.loss}
        val_num_foldings, val_log_jac_det_std, val_tre, val_dice = [], [], [], []
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
                (
                    batch_num_foldings,
                    batch_log_jac_det_std,
                    batch_tre,
                    batch_dice,
                ) = self.__compute_metrics(
                    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
                )
                val_num_foldings.extend(batch_num_foldings)
                val_log_jac_det_std.extend(batch_log_jac_det_std)
                val_tre.extend(batch_tre)
                val_dice.extend(batch_dice)

                val_loss, val_all_loss = self.__compute_loss(
                    self.loss_fn, fixed_img, moving_reg, fixed_mask, moving_mask_reg, rf
                )
                val_loss_sum += val_loss.item()
                for l , loss_value in val_sub_loss_sum.items():
                    val_sub_loss_sum[l] = (
                        loss_value + val_all_loss[l]
                    )


        val_loss_mean = val_loss_sum / len(val_loader)
        val_sub_loss_mean = {l: loss_value / len(val_loader) for l, loss_value in val_sub_loss_sum.items()}

        val_num_foldings_mean = np.mean(val_num_foldings)
        val_log_jac_det_std_mean = np.mean(val_log_jac_det_std)
        val_tre_mean = np.mean(val_tre)
        val_dice_mean = np.mean(val_dice)
        print("Epoch {} - Validation Loss : {:.6f}".format(cur, val_loss_mean))
        for l, vslm in val_sub_loss_mean.items():
            print('\t\t |- {} loss: {:.6f}'.format(l, vslm))
        print(
            "\t\tDice: {:.6f}; TRE: {:.6f}; NumFold: {:.6f}; LogJacDetStd: {:6f}".format(
                val_dice_mean,
                val_tre_mean,
                val_num_foldings_mean,
                val_log_jac_det_std_mean,
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

    def predict(self, val_loader: torch.utils.data.DataLoader) -> pd.DataFrame:
        # load final model
        print("-------Load Final Model Checkpoint-------")
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

        val_loss_sum = 0
        val_sub_loss_sum = {l: 0.0 for l in self.loss}
        val_num_foldings, val_log_jac_det_std, val_tre, val_dice = [], [], [], []
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
                (
                    batch_num_foldings,
                    batch_log_jac_det_std,
                    batch_tre,
                    batch_dice,
                ) = self.__compute_metrics(
                    D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
                )
                val_num_foldings.extend(batch_num_foldings)
                val_log_jac_det_std.extend(batch_log_jac_det_std)
                val_tre.extend(batch_tre)
                val_dice.extend(batch_dice)

                val_loss, val_all_loss = self.__compute_loss(
                    self.loss_fn, fixed_img, moving_reg, fixed_mask, moving_mask_reg, rf
                )
                val_loss_sum += val_loss.item()
                for l , loss_value in val_sub_loss_sum.items():
                    val_sub_loss_sum[l] = (
                        loss_value + val_all_loss[l]
                    )

                # save displacement field - need to double check with learn2reg
                if self.save_df:
                    os.makedirs(
                        os.path.join(self.exp_dir, "displacement_field"), exist_ok=True
                    )
                    for iidx, subject in enumerate(
                        val_loader.dataset.subjects[
                            batch_idx
                            * val_loader.batch_size : (batch_idx + 1)
                            * val_loader.batch_size
                        ]
                    ):
                        subject_id = subject["moving"].split("/")[-1].split("_")[1]
                        np.save(
                            os.path.join(
                                self.exp_dir,
                                "displacement_field",
                                "disp_NLST_{}.npy".format(
                                    subject_id
                                ),  # need to change when doing another task
                            ),
                            D_rf[iidx].permute(1, 2, 3, 0).detach().cpu().numpy(),
                        )

        val_loss_mean = val_loss_sum / len(val_loader)
        val_sub_loss_mean = {l: vsls / len(val_loader) for l, vsls in val_sub_loss_sum.items()}
        val_num_foldings_mean = np.mean(val_num_foldings)
        val_log_jac_det_std_mean = np.mean(val_log_jac_det_std)
        val_tre_mean = np.mean(val_tre)
        val_dice_mean = np.mean(val_dice)

        print("Final validation Loss : {:.6f}".format(val_loss_mean))
        for l, vslm in val_sub_loss_mean.items():
            print('\t\t |- {} loss : {:.6f}'.format(l, vslm))
        print(
            "\t\tDice: {:.6f}; TRE: {:.6f}; NumFold: {:.6f}; LogJacDetStd: {:6f}".format(
                val_dice_mean,
                val_tre_mean,
                val_num_foldings_mean,
                val_log_jac_det_std_mean,
            )
        )

        # save results
        results = pd.DataFrame(
            {
                "val": [
                    val_loss_mean,
                    val_dice_mean,
                    val_tre_mean,
                    val_num_foldings_mean,
                    val_log_jac_det_std_mean,
                ],
            },
            index=["loss", "dice", "tre", "num_fold", "log_jac_det_std"],
        )
        results_sub_loss = pd.DataFrame(
            val_sub_loss_mean, index=["val"]
        ).T
        results = pd.concat([results, results_sub_loss], axis=0)

        return results

    @staticmethod
    def __compute_metrics(
        D_rf, fixed_kp, moving_kp, fixed_mask, moving_mask, moving_mask_reg
    ):
        batch_num_foldings, batch_log_jac_det_std, batch_tre, batch_dice = [], [], [], []

        for subject_idx in range(len(D_rf)):
            # jacobian determinant
            num_foldings, log_jac_det_std = jacobian_determinant(
                D_rf[subject_idx : subject_idx + 1].clone().detach().cpu().numpy()
            )
            
            # TRE keypoints
            tre = compute_tre(
                fix_lms=fixed_kp[subject_idx].clone().detach().cpu().numpy(),
                mov_lms=moving_kp[subject_idx].clone().detach().cpu().numpy(),
                disp=D_rf.permute(0, 2, 3, 4, 1)[subject_idx]
                .clone()
                .detach()
                .cpu()
                .numpy(),
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
            batch_num_foldings.append(num_foldings)
            batch_log_jac_det_std.append(log_jac_det_std)
            batch_tre.append(tre)
            batch_dice.append(mean_dice)
        return batch_num_foldings, batch_log_jac_det_std, batch_tre, batch_dice

    @staticmethod
    def __compute_loss(loss_fn, fixed_img, moving_reg, fixed_mask, moving_mask_reg, rf):
        loss = 0.0
        all_loss = {}
        for l, lw in loss_fn.items():
            if l == "NCC":
                ncc_loss = lw[0](fixed_img, moving_reg)
                loss += lw[1] * ncc_loss
                all_loss["NCC"] = ncc_loss.item()
            elif l == "Smooth":
                smooth_loss = lw[0](rf)
                loss += lw[1] * smooth_loss
                all_loss["Smooth"] = smooth_loss.item()
            elif l == "Dice":
                dice_loss = lw[0](fixed_mask, moving_mask_reg)
                loss += lw[1] * dice_loss
                all_loss["Dice"] = dice_loss.item()
            elif l == "MSE":
                mse_loss = lw[0](fixed_img, moving_reg)
                loss += lw[1] * mse_loss
                all_loss["MSE"] = mse_loss.item()
            elif l == "SAD":
                sad_loss = lw[0](fixed_img, moving_reg)
                loss += lw[1] * sad_loss
                all_loss["SAD"] = sad_loss.item()

        return loss, all_loss
