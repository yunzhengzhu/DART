from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import *

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
        return x


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
        self.head = DoubleConv(cnn_encoder_channels[-1], in_channels//2)
        self.vit = ViT(
            image_size=image_size,
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels//2,
        )
        
        self.decoder = DecoderCup(
            in_channels//2,
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
        x = self.head(cnn_output)
        x = self.vit(x)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        return flow
