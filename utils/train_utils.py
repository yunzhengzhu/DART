import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self) -> None:
        pass
