from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import *

# 3D MAE
class MAE(nn.Module):
    def __init__(
        self, 
        image_size=(112,96,112), 
        patch_size=16,
        encoder_dim=768,
        mlp_dim=3072,
        channels=1,
        decoder_dim=512,
        masking_ratio = 0.75,
    ):
        super(MAE, self).__init__()
        self.masking_ratio = masking_ratio

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
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters
        self.mask_token = nn.Parameter(torch.randn(encoder_dim))
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
        
        # calculate patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * n_patches)
        rand_indices = torch.rand(batch, n_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:] # [B, num_masked_ind] [B, num_unmasked_ind]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices] # [B, num_unmasked_ind, encoder_dim]
        
        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices] # [B, num_masked_ind, patch_dim]
        
        # attend with vision transformer
        encoded_tokens = self.Transformer_encoder(tokens) # [B, num_unmasked_ind, encoder_dim]

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked) # [B, num_masked_ind, encoder_dim]
        encoder_tokens = torch.cat((encoded_tokens, mask_tokens), dim=1) # [B, num_patches, encoder_dim]
        full_tokens = encoder_tokens
        encoder_tokens = encoder_tokens.transpose(1, 2)
        H, W, D = img.shape[2:]
        cuberoot = [H // self.patch_size, W // self.patch_size, D // self.patch_size]
        x_shape = encoder_tokens.size()
        x = torch.reshape(encoder_tokens, [x_shape[0], x_shape[1], cuberoot[0], cuberoot[1], cuberoot[2]])

        x = self.conv3d_transpose(x) # [1, 512, 28, 24, 28]

        mask_tokens = full_tokens[:, -num_masked:]
        
        return x


class MAE_Finetune_Baseline(nn.Module):
    def __init__(
        self, 
        in_channels=2,
        image_size=(112, 96, 112),
        cnn_encoder_channels=(16, 32, 32), #(128, 256, 256), #(16, 32, 32),
        skip_channels=(32, 32, 32, 32, 16), #(256, 256, 256, 256, 128), #(32, 32, 32, 32, 16),
        decoder_channels=(96, 48, 32, 32, 16), #(768, 384, 256, 256, 128), #(96, 48, 32, 32, 16),
        n_skips=5,
        reg_in_channels=16, #128,
        reg_out_channels=3,
        down_factor=2,
    ):
        super(MAE_Finetune_Baseline, self).__init__()
        self.cnn_encoder = CNNEncoder(n_channels=in_channels, encoder_channels=cnn_encoder_channels, n_skips=n_skips)
        self.head = DoubleConv(cnn_encoder_channels[-1], in_channels//2)
        self.mae_transformer = MAE(
            image_size=(image_size[0]//2**down_factor, image_size[1]//2**down_factor, image_size[2]//2**down_factor),
            patch_size=16//2**down_factor,
            encoder_dim=768,
            mlp_dim=3072,
            channels=in_channels//2,
            masking_ratio=0.0, #0.75,
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

    def forward(self, x, y, x_kp=None):
        cnn_output, features = self.cnn_encoder(
            torch.cat((x, y), dim=1)
        )
        x = self.head(cnn_output)
        x = self.mae_transformer(x)
        x = self.decoder(x, features)
        flow = self.reg_head(x)
        return flow
