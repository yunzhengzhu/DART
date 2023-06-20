import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import time
from argparse import ArgumentParser
from utils.train_utils import EarlyStopping


class baseTrainer:
    def __init__(self, args: ArgumentParser) -> None:
        self.args = args
        self.loss = self.__init_loss()
        self.optimizer = self.__init_optimizer()
        self.model = self.__init_model()
        self.device = self.__get_device()
        if args.es:
            # TODO early stopping
            self.early_stopping = EarlyStopping(patience=10, verbose=True)

    def __init_optimizer(self):
        pass

    def __init_loss(self):
        pass

    def __init_model(self):
        pass

    @staticmethod
    def __get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(baseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.epochs = args.epochs

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        # loop through epochs
        self.model.train()
        for i in range(self.epochs):
            # training
            train_loss_sum = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                # TODO data to device
                # TODO pass data to model
                output = self.model()
                train_loss = self.loss(output)
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
            for batch_idx, (data, target) in enumerate(val_loader):
                # TODO data to device
                # TODO pass data to model
                output = self.model(val_loader)
                val_loss = self.loss(output)
                val_loss_sum += val_loss.item()
        val_loss_avg = val_loss_sum / len(val_loader)

        print("Validation Loss: {:.6f}".format(val_loss_avg))
        # TODO compute metrics

        # early stopping
        if self.args.es:
            earlystop = self.early_stopping(val_loss_avg, self.model)
            return earlystop
        else:
            return False

    def predict(self, val_loader):
        # TODO
        pass

    def save(self):
        pass
