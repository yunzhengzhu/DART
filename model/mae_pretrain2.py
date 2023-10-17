import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        inner_dim = dim_head * heads
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
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
        )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x

class MAE(nn.Module):
    def __init__(
        self,
        image_size=(112,96,112),
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        channels=1,
        decoder_dim=512,
        masking_ratio=0.75,
        decoder_depth=6,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        num_patches = (image_size[0] // patch_size) * \
                      (image_size[1] // patch_size) * \
                      (image_size[2] // patch_size)
        
        patch_dim = channels * patch_size * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_embeddings = nn.Sequential(
            Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1 = patch_size, p2 = patch_size, p3 = patch_size),
            nn.Linear(patch_dim, encoder_dim)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, encoder_dim))
        self.to_patch, self.patch_to_emb = self.patch_embeddings[:2]
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.Transformer_encoder = Transformer(
            dim=encoder_dim, 
            depth=12, 
            heads=12, 
            dim_head=decoder_dim_head, 
            mlp_dim=mlp_dim, #decoder_dim * 4, 
            dropout=0.1
        )
        self.Transformer_decoder = Transformer(
            dim=decoder_dim, 
            depth=decoder_depth, 
            heads=decoder_heads, 
            dim_head=decoder_dim_head, 
            mlp_dim=decoder_dim*4, 
            dropout=0.1
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        new_patch_size = (4, 4, 4)
        self.conv3d_transpose = nn.ConvTranspose3d(decoder_dim, out_channels=channels, kernel_size=new_patch_size, stride=new_patch_size)
        #self.conv3d_transpose_1 = nn.ConvTranspose3d(
        #    in_channels=16, out_channels=channels, kernel_size=new_patch_size, stride=new_patch_size
        #)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        """
        img: (B, C, H, W, D)

        """
        device = img.device
        ##################### Preparation: get patches #######################

        patches = self.to_patch(img)    # [B, num_patches=(H//patch_size) * (W//patch_size) * (D//patch_size), patch_dim=(C*H*D*W)]
        batch, num_patches, *_ = patches.shape
        
        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches) # [B/2, num_patches, encoder_dim]
        tokens = tokens + self.pos_embedding[:, :(num_patches)] # [B/2, num_patches, encoder_dim]

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches) # num_masked_patches=mask_ratio * num_patches
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1) # [B, num_patches]
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:] # [B, num_masked_ind] [B, num_unmasked_ind]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None] # [B, 1]
        tokens = tokens[batch_range, unmasked_indices]
        #tokens = torch.cat([torch.index_select(t, 0, umidx).unsqueeze(0) for t, umidx in zip(tokens, unmasked_indices)]) # [B, num_unmasked_ind, encoder_dim]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        #masked_patches = torch.cat([torch.index_select(p, 0, midx).unsqueeze(0) for p, midx in zip(patches, masked_indices)]) # [B, num_masked_ind, patch_dim]

        ####################################################################
        # attend with vision transformer

        # encoder to decoder
        tokens = self.Transformer_encoder(tokens) # Same as before [B, num_unmasked_ind, encoder_dim]

        decoder_tokens = self.enc_to_dec(tokens) # [B, num_unmasked_ind, decoder_dim]
        
        #decoder_tokens = self.to_latent(decoder_tokens) # [B, num_unmasked_ind, decoder_dim]
        #decoder_tokens = self.mlp_head(decoder_tokens) # [B, num_unmasked_ind, decoder_dim]

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        
        # reapply decoder position embedding to unmasked tokens

        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices) # [B, num_unmasked_ind, decoder_dim]

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked) # [B, num_masked_ind, decoder_dim]
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices) # [B, num_masked_ind, decoder_dim]
        
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1) # [B, num_patches, decoder_dim]
        decoded_tokens = self.Transformer_decoder(decoder_tokens) # [B, num_patches, decoder_dim]

        decoder_tokens = decoder_tokens.transpose(1, 2) # [B, decoder_dim, num_patches]
        #cuberoot = round(math.pow(decoder_tokens.size()[2], 1/3)) # round(num_patches ** 1/3)
        H, W, D = img.shape[2:]
        cuberoot = [H // self.patch_size, W // self.patch_size, D // self.patch_size]
        x_shape = decoder_tokens.size() # B, decoder_dim, num_patches
        x = torch.reshape(decoder_tokens, [x_shape[0], x_shape[1], cuberoot[0], cuberoot[1], cuberoot[2]]) # [B, decoder_dim, cr, cr, cr]
        x = self.conv3d_transpose(x) # [B, out_c(16), cr * stride(4), cr * stride(4), cr * stride(4)] 
        #x = self.conv3d_transpose_1(x) # [B, out_c(1), cr * (stride(4) ** 2), cr * (stride(4) ** 2), cr * (stride(4) ** 2)]

        # splice out the mask tokens and project to pixel values
        #mask_tokens = decoded_tokens[:, :num_masked]
        mask_tokens = decoded_tokens[:, -num_masked:] # [B, num_masked_ind, decoder_dim]
        #pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        #recon_loss = F.mse_loss(self.to_pixels(mask_tokens), masked_patches)
        #return x, recon_loss
        return x, self.to_pixels(mask_tokens), masked_patches

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
        self.se_3D_layer = ChannelSpatialSELayer3D(encoder_channels[2])

    def forward(self, x):
        features = []
        x1 = self.inc(x)
        features.append(x1)
        x2 = self.down1(x1)
        features.append(x2)
        x3 = self.down2(x2)
        x3 = self.se_3D_layer(x3)
        features.append(x3)
        feats_down = x3
        for i in range(self.down_num):
            feats_down = nn.MaxPool3d(2)(feats_down)  # 3D MaxPooling
            features.append(feats_down)
        return x3, features[::-1]

class MAE_Pretrain(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels=1,
        down_factor=2,
    ):
        super().__init__()
        self.mae_transformer1 = MAE(
            image_size=image_size,
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels//2,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )
        self.mae_transformer2 = MAE(
            image_size=image_size,
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels//2,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )

    def forward(self, x, y):
        output1, recon1, orig1 = self.mae_transformer1(x)
        output2, recon2, orig2 = self.mae_transformer2(y)
        return output1, output2, recon1, recon2, orig1, orig2

class MAE_Pretrain_HybridNet(nn.Module):
    def __init__(
        self,
        image_size=(112, 96, 112),
        in_channels=1,
        cnn_encoder_channels=(16, 32, 32),
        n_skips=5,
        down_factor=2
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoder(n_channels=in_channels*2, encoder_channels=cnn_encoder_channels, n_skips=n_skips)
        self.head = DoubleConv(cnn_encoder_channels[-1], in_channels)

        self.mae_transformer = MAE(
            image_size=(image_size[0]//2**down_factor, image_size[1]//2**down_factor, image_size[2]//2**down_factor),
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )

    def forward(self, x):
        #print(f"x:{x.shape}")
        cnn_output, _ = self.cnn_encoder(
            torch.cat((x, x), dim=1)
        )
        #print(f"cnn_output:{cnn_output.shape}")
        x = self.head(cnn_output)
        #print(f"head:{x.shape}")
        output, recon, orig = self.mae_transformer(x)
        #print(f"output: {output.shape}")
        return output, recon, orig


class MAE_Pretrain_Baseline(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels=1,
        down_factor=2,
    ):
        super().__init__()
        self.mae_transformer = MAE(
            image_size=image_size,
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels,
            decoder_dim=512,
            masking_ratio=0.75,
            decoder_depth=6,
            decoder_heads=8,
            decoder_dim_head=64
        )

    def forward(self, x):
        #print(f"x:{x.shape}")
        output, recon, orig = self.mae_transformer(x)
        #print(f"output:{output.shape}")
        return output, recon, orig


# mae = MAE(
#     image_size=(64,128,128),
#     patch_size=16,
#     encoder_dim=768,
#     mlp_dim=3072,
#     masking_ratio = 0.75,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )
#
# images = torch.randn(1,1,64,128,128)
# print(mae)
# Output,loss = mae(images)
#
# loss.backward()
# print("Output_shape:",Output.shape)
# print("Loss:",loss)
#
# # that's all!
# # do the above in a for loop many times with a lot of images and your vision transformer will learn
#
# # save your improved vision transformer
# torch.save(v.state_dict(), './trained-vit.pt')

# from thop import profile, clever_format
# net = MAE(
#     image_size=(64,128,128),
#     patch_size=16,
#     encoder_dim=768,
#     mlp_dim=3072,
#     masking_ratio = 0.75,   # the paper recommended 75% masked patches
#     decoder_dim = 512,      # paper showed good results with just 512
#     decoder_depth = 6       # anywhere from 1 to 8
# )
# flops, params = profile(net, inputs=(images,))
# macs, params = clever_format([flops, params], "%.3f") # 格式化输出
# print('params:',params) # 模型参数量

'''
MAE-base
MAE(
    image_size=(64,128,128),
    patch_size=16,
    encoder_dim=768,
    mlp_dim=3072,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)

MAE-large
MAE(
    image_size=(64,128,128),
    patch_size=16,
    encoder_dim=1024,
    mlp_dim=4096,
    masking_ratio = 0.125,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
)
'''
