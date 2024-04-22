from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.nn.functional as F
from model.model_utils import *

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
        decoder_dim_head=64,
        num_masks=1
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
            mlp_dim=mlp_dim, 
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
        self.conv3d_transpose_seg = nn.ConvTranspose3d(decoder_dim, out_channels=num_masks, kernel_size=new_patch_size, stride=new_patch_size)
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

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]

        ####################################################################
        # attend with vision transformer

        # encoder to decoder
        tokens = self.Transformer_encoder(tokens) # Same as before [B, num_unmasked_ind, encoder_dim]

        decoder_tokens = self.enc_to_dec(tokens) # [B, num_unmasked_ind, decoder_dim]
        
        # reapply decoder position embedding to unmasked tokens
        decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices) # [B, num_unmasked_ind, decoder_dim]

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked) # [B, num_masked_ind, decoder_dim]
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices) # [B, num_masked_ind, decoder_dim]
        
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1) # [B, num_patches, decoder_dim]
        decoded_tokens = self.Transformer_decoder(decoder_tokens) # [B, num_patches, decoder_dim]

        decoder_tokens = decoder_tokens.transpose(1, 2) # [B, decoder_dim, num_patches]

        H, W, D = img.shape[2:]
        cuberoot = [H // self.patch_size, W // self.patch_size, D // self.patch_size]
        x_shape = decoder_tokens.size() # B, decoder_dim, num_patches
        x = torch.reshape(decoder_tokens, [x_shape[0], x_shape[1], cuberoot[0], cuberoot[1], cuberoot[2]]) # [B, decoder_dim, cr, cr, cr]

        # recon head
        x_recon = self.conv3d_transpose(x) # [B, out_c(16), cr * stride(4), cr * stride(4), cr * stride(4)] 

        # seg head
        x_seg = self.conv3d_transpose_seg(x)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, -num_masked:] # [B, num_masked_ind, decoder_dim]

        return x_recon, x_seg, self.to_pixels(mask_tokens), masked_patches


class MAE_Pretrain_SegNet(nn.Module):
    def __init__(
        self,
        image_size=(112, 96, 112),
        in_channels=1,
        n_skips=5,
        down_factor=2,
        num_masks=1,
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
            decoder_dim_head=64,
            num_masks=num_masks,
        )

    def forward(self, x):
        output_recon, output_seg, recon, orig = self.mae_transformer(x)
        return output_recon, output_seg, recon, orig
