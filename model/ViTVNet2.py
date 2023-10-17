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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads # 16 * 8 = 128
        self.heads = heads
        self.scale = dim ** -0.5
        self.se_layer = SELayer(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        x = self.se_layer(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
            # nn.Linear(dim, 512)
        )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


# 3D ViT
class ViT(nn.Module):
    def __init__(
        self, 
        image_size=(112,96,112), 
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        channels=1,
    ):
        super(ViT, self).__init__()

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)
        num_patches = (image_size[0] // patch_size) * \
                      (image_size[1] // patch_size) * \
                      (image_size[2] // patch_size)
        
        patch_dim = channels * patch_size * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embeddings = nn.Sequential(
            Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=patch_size, p2=patch_size, p3=patch_size),
            nn.Linear(patch_dim, encoder_dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, encoder_dim))
        self.to_patch, self.patch_to_emb = self.patch_embeddings[:2]

        self.Transformer_encoder = Transformer(
            dim=encoder_dim, 
            depth=12, 
            heads=12, 
            dim_head=64, 
            mlp_dim=mlp_dim, 
            dropout=0.1
        )
        new_patch_size = (4, 4, 4)
        self.new_patch_size = new_patch_size
        self.conv3d_transpose = nn.ConvTranspose3d(encoder_dim, channels, kernel_size=new_patch_size, stride=new_patch_size)
        #self.conv3d_transpose_1 = nn.ConvTranspose3d(
        #    in_channels=16, out_channels=1, kernel_size=new_patch_size, stride=new_patch_size
        #)

    def forward(self, img):
        device = img.device
       
       # get patches
        patches = self.to_patch(img) # [B, num_patches=(H//patch_size) * (W//patch_size) * (D//patch_size), patch_dim=(C*H*D*W)]

        batch, n_patches, *_ = patches.shape
         
        # patch to encoder tokens ha add positions
        tokens = self.patch_to_emb(patches)  # [B, num_patches, encoder_dim] 
        tokens += self.pos_embedding[:, :(n_patches)] # [B, num_patches, encoder_dim]
        
        # attend with vision transformer
        encoded_tokens = self.Transformer_encoder(tokens)

        encoded_tokens = encoded_tokens.transpose(1, 2)
        H, W, D = img.shape[2:]
        cuberoot = [H // self.patch_size, W // self.patch_size, D // self.patch_size]
        x_shape = encoded_tokens.size()
        x = torch.reshape(encoded_tokens, [x_shape[0], x_shape[1], cuberoot[0], cuberoot[1], cuberoot[2]]) # [1, 512, 294] -> [1, 512, 7, 6, 7]
        x = self.conv3d_transpose(x) # [1, 512, 28, 24, 28]
        #x = self.conv3d_transpose_1(x)

        return x


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


class ViTVNet(nn.Module):
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
        down_factor=2
    ):
        super(ViTVNet, self).__init__()
        self.cnn_encoder = CNNEncoder(n_channels=in_channels, encoder_channels=cnn_encoder_channels, n_skips=n_skips)
        #self.head = DoubleConv(cnn_encoder_channels[-1], 1)
        self.vit = ViT(
            image_size=image_size,
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=cnn_encoder_channels[-1],
        )
        
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
        x = self.vit(cnn_output)
        #print(f"x:{x.shape}") # 1, 512, 18816
        
        x = self.decoder(x, features)
        #print(f"decoder out x:{x.shape}")
        flow = self.reg_head(x)

        return flow


