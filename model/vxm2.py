from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, normal_



import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
from torch.distributions.normal import Normal

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = nnf.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor

class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None, up=True):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.up(x) if up else x
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderCup(nn.Module):
    def __init__(
            self, 
            hidden_size,
            img_size=(112, 96, 112), 
            patch_size=4, 
            n_skips=5, 
            head_channels=512,
            decoder_channels=(96, 48, 32, 32, 16),
            skip_channels=(32, 32, 32, 32, 16)
        ):
        super().__init__()
        self.down_factor = 2 #config.down_factor
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            hidden_size, #config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(patch_size)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.n_skips = n_skips

    def forward(self, hidden_states, features=None):
        x = hidden_states
        x = F.interpolate(x, scale_factor=1/2**self.down_factor, mode='trilinear', align_corners=False)
        x = self.conv_more(x)
        up = True
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skips) else None
                #print(skip.shape)
            else:
                skip = None
            #if i == self.down_factor+1:
            #    x = F.interpolate(x, scale_factor=1/2**self.down_factor, mode='trilinear', align_corners=False)
            if i == len(self.blocks) - 1:
                up = False
            x = decoder_block(x, skip=skip, up=up)
        return x


class saliency_map_attention_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels):
        super().__init__()

        self.att_block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1=self.att_block(x)
        x2=torch.mul(x,x1)
        return torch.add(x,x2)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            saliency_map_attention_block(out_channels),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            saliency_map_attention_block(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class CNNEncoder(nn.Module):
    def __init__(self, encoder_channels, n_channels=2, n_skips=4):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        self.down_num = n_skips - 3 #2 #config.down_num
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        self.down2 = Down(encoder_channels[1], encoder_channels[2])
        #self.up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.se_3D_layer = ChannelSpatialSELayer3D(encoder_channels[2])

    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        x3 = self.down2(x2)
        #x3 = self.up(x3)
        x3 = self.se_3D_layer(x3)
        features.append(x3)
        feats_down = x3
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)  # 3D MaxPooling
            features.append(feats_down)
        return x3, features[::-1]

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class MAE_Finetune(nn.Module):
    def __init__(
        self, 
        image_size=(112, 96, 112),
        cnn_encoder_channels=(16, 32, 32),
        n_skips=5,
    ):
        super(MAE_Finetune, self).__init__()
        self.mae_transformer1 = MAE(
            image_size=image_size,
            patch_size=16,
            encoder_dim=768,
            mlp_dim=3072,
            channels=1,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )
        self.mae_transformer2 = MAE(
            image_size=image_size,
            patch_size=16,
            encoder_dim=768,
            mlp_dim=3072,
            channels=1,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )
        self.cnn_encoder = CNNEncoder(n_channels=2, encoder_channels=cnn_encoder_channels, n_skips=n_skips)
        self.decoder = DecoderCup(img_size=image_size, patch_size=4, n_skips=n_skips)
        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        

    def forward(self, x, y):
        print(f"x:{x.shape}")
        print(f"y:{y.shape}")
        x1, recon_loss1 = self.mae_transformer1(x)
        x2, recon_loss2 = self.mae_transformer2(y)
        print(f"x1:{x1.shape}")
        print(f"x2:{x2.shape}")
        cnn_output, features = self.cnn_encoder1(
            torch.cat((x1, x2), dim=1),
        )
        print(f"cnn_output:{cnn_output.shape}")
        print([f.shape for f in features]) 
        x = self.decoder(cnn_output, features)
        print(f"x:{x.shape}")
        flow = self.reg_head(x)

        return flow

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
        #self.head = DoubleConv(cnn_encoder_channels[-1], 1)
        
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
        #print(f"x:{x.shape}")
        #print(f"y:{y.shape}")
        cnn_output, features = self.cnn_encoder(
            torch.cat((x, y), dim=1)
        )
        #print(f"cnn_output:{cnn_output.shape}")
        
        #x = self.head(cnn_output)
        #print(f"head:{x.shape}")
        
        #print([f.shape for f in features])
        #print(f"x:{x.shape}") # 1, 512, 18816
        
        x = self.decoder(cnn_output, features)
        #print(f"decoder out x:{x.shape}")
        flow = self.reg_head(x)

        return flow


