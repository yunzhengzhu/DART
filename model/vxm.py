from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import *

class Voxelmorph(nn.Module):
    def __init__(
        self,
        in_channels=2,
        image_size=(112, 96, 112),
        cnn_encoder_channels=(16, 32, 32),
        skip_channels=(32, 32, 32, 32, 16),
        decoder_channels=(96, 48, 32, 32, 16),
        n_skips=5,
        reg_in_channels=16,
        reg_out_channels=3,
    ):
        super(Voxelmorph, self).__init__()
        self.cnn_encoder = CNNEncoder(n_channels=in_channels, encoder_channels=cnn_encoder_channels, n_skips=n_skips)
        
        self.decoder = DecoderCup(
            cnn_encoder_channels[-1],
            img_size=image_size, 
            patch_size=4, 
            n_skips=n_skips, 
            head_channels=512,
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
        )
        
        self.reg_head = RegistrationHead(
            in_channels=reg_in_channels,
            out_channels=reg_out_channels,
            kernel_size=3,
        )
        

    def forward(self, x, y):
        cnn_output, features = self.cnn_encoder(
            torch.cat((x, y), dim=1)
        )
        x = self.decoder(cnn_output, features)
        flow = self.reg_head(x)
        return flow
