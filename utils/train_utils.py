import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    def __init__(
        self, warmup: int = 10, patience: int = 20, verbose: bool = True
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.lowest_val_loss = np.Inf
        self.early_stop = False
        self.warmup = warmup

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        model: torch.nn.Module,
        ckpt_path: str = "checkpoint.pt",
    ) -> None:
        if epoch < self.warmup:
            pass
        elif np.isinf(self.lowest_val_loss):
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
        elif val_loss > self.lowest_val_loss:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.lowest_val_loss = val_loss
            self.counter = 0

    def save_checkpoint(
        self, val_loss: float, model: torch.nn.Module, ckpt_path: str
    ) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased from {self.lowest_val_loss:.6f} to {val_loss:.6f}. Model saved."
            )
        torch.save({"model": model.state_dict()}, ckpt_path)
