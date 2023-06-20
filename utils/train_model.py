import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import time
from argparse import ArgumentParser
import model.lkunet.UNet as LKUNet
from model.transform import SpatialTransform, DiffeomorphicTransform
from utils.loss_utils import smoothLoss, NCC, Dice, MSE, SAD
from utils.train_utils import EarlyStopping

class baseTrainer():
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

        self.__init_sim_loss()
        self.__init_smooth_loss()
        self.__init_optimizer()
        self.__init_model()
        self.__get_device()

        if args.es:
            self.early_stopping = EarlyStopping(warmup=self.es_warmup, tolerence=self.es_tolerence, verbose =True)
        else:
            self.early_stopping = None

    def __init_optimizer(self):
        print(f'Initiate {self.opt} optimizer')

        if self.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def __init_sim_loss(self):
        if self.loss == "NCC":
            self.sim_loss = NCC()
        elif self.loss == "Dice":
            self.sim_loss = Dice()
        elif self.loss == "SAD":
            self.sim_loss = SAD()
        elif self.loss == "MSE":
            self.sim_loss = MSE()
        
    def __init_smooth_loss(self):
        self.smooth_loss = smoothLoss()

    def __init_model(self):
        if self.model_type == "LKU-Net":
            self.model = LKUNet()
        else:
            raise NotImplementedError

        self.spatial_transform = SpatialTransform()
        self.diff_transform = DiffeomorphicTransform()

        self.model.to(self.device)
        self.spatial_transform.to(self.device)
        self.diff_transform.to(self.device)

    def __get_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(baseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.epochs = args.epochs
        self.smooth_w = args.smooth_w
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ):
        # loop through epochs
        self.model.train()
        for i in range(self.epochs):
            # training
            train_loss_sum = 0
            for batch_idx, (fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask) in enumerate(train_loader):
                fixed_img, moving_img = fixed_img.to(self.device), moving_img.to(self.device)
                rf = self.model(moving_img, fixed_img)
                D_rf = self.diff_transform(rf)
                moving_reg = self.spatial_transform(fixed_img, D_rf.permute(0, 2, 3, 4, 1))
                
                train_loss = self.sim_loss(fixed_img, moving_reg) + self.smooth_w * self.smooth_loss(rf) 
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss_sum += train_loss.item()

            train_loss_avg = train_loss_sum / len(train_loader)
            print("Epoch: {} \tTraining Loss: {:.6f}".format(i, train_loss_avg))
            # TODO compute metrics

            # validation
            earlystop = self.__eval(val_loader)
            if earlystop:
                break

    def __eval(self, val_loader: torch.utils.data.DataLoader):
        val_loss_sum = 0
        with torch.no_grad():
            for batch_idx, (fixed_img, moving_img, fixed_kp, moving_kp, fixed_mask, moving_mask) in enumerate(val_loader):
                fixed_img, moving_img = fixed_img.to(self.device), moving_img.to(self.device)
                fixed_mask, moving_mask = fixed_mask.to(self.device), moving_mask.to(self.device)

                #pass data to model
                rf = self.model(moving_img, fixed_img)
                D_rf = self.diff_transform(rf)
                moving_reg = self.spatial_transform(fixed_img, D_rf.permute(0, 2, 3, 4, 1))

                val_loss = self.sim_loss(fixed_img, moving_reg) + self.smooth_w * self.smooth_loss(rf) 
                val_loss_sum += val_loss.item()
        val_loss_avg = val_loss_sum / len(val_loader)

        print("Validation Loss: {:.6f}".format(val_loss_avg))
        # TODO compute metrics

        # early stopping
        if self.es:
            earlystop = self.early_stopping(val_loss_avg, self.model)
            return earlystop
        else:
            return False

    def predict(self, val_loader):
        # TODO
        pass

    def save(self):
        pass
