import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=[1, 1]):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels,out_channels,3,padding=1,bias=False,groups=groups[0]),\
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels,out_channels,1,bias=False,groups=groups[1]),\
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels,2,stride=2,padding=0,output_padding=0,bias=bias),\
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.deconv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self,F_g,F_l,F_int,bias=False):
        super(AttentionBlock,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=bias),
            nn.BatchNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=bias),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=bias),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class LK_encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        kernel_size_large=7,
        stride=1,
        padding=2,
        bias=False,
        norm='batchnorm',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_large = kernel_size_large
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.norm = norm

        super(LK_encoder, self).__init__()

        self.layer_regularKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
            norm=self.norm,
        )
        self.layer_largeKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            norm=self.norm,
        )
        #self.layer_largelargeKernel = self.encoder_LK_encoder(
        #    self.in_channels,
        #    self.out_channels,
        #    kernel_size=self.kernel_size_large,
        #    stride=self.stride,
        #    padding=self.padding+1,
        #    bias=self.bias,
        #    norm=self.norm,
        #)
        self.layer_oneKernel = self.encoder_LK_encoder(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
            norm=self.norm,
        )
        self.layer_nonlinearity = nn.PReLU()
        # self.layer_batchnorm = nn.BatchNorm3d(num_features = self.out_channels)

    def encoder_LK_encoder(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        norm=False,
    ):
        if norm == "batchnorm":
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.BatchNorm3d(out_channels),
            )
        elif norm == "instancenorm":
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                ),
                nn.InstanceNorm3d(out_channels),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        return layer


    def forward(self, inputs):
        # print(self.layer_regularKernel)
        regularKernel = self.layer_regularKernel(inputs)
        largeKernel = self.layer_largeKernel(inputs)
        #largelargeKernel = self.layer_largelargeKernel(inputs)
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        #outputs = regularKernel + largelargeKernel + largeKernel + oneKernel + inputs
        outputs = regularKernel + largeKernel + oneKernel + inputs
	# else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
        # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)


class AACN_Layer(nn.Module):
    def __init__(self, in_channels, k=0.25, v=0.25, kernel_size=3, num_heads=8, image_size=(224, 192, 224), positional_encoding=None, inference=False):
        super(AACN_Layer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = math.floor((in_channels*k)/num_heads)*num_heads 
        # Paper: A minimum of 20 dimensions per head for the keys
        if self.dk / num_heads < 20:
            self.dk = num_heads * 20
        self.dv = math.floor((in_channels*v)/num_heads)*num_heads
        
        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads. (example: dk: 32, num_heads: 8)"
        assert self.dv % self.num_heads == 0, "dv should be divided by num_heads. (example: dv: 32, num_heads: 8)"  
        
        self.padding = (self.kernel_size - 1) // 2
        
        self.conv_out = nn.Conv3d(self.in_channels, self.in_channels - self.dv, self.kernel_size, padding=self.padding).to(self.device)
        self.kqv_conv = nn.Conv3d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        self.attn_out = nn.Conv3d(self.dv, self.dv, 1).to(self.device)
        
        # Positional encodings
        self.positional_encoding = positional_encoding
        if self.positional_encoding != None:
            self.rel_encoding_h = nn.Parameter(torch.randn((2 * image_size[0] - 1, self.dk // self.num_heads), requires_grad=True))
            self.rel_encoding_w = nn.Parameter(torch.randn((2 * image_size[1] - 1, self.dk // self.num_heads), requires_grad=True))
            self.rel_encoding_d = nn.Parameter(torch.randn((2 * image_size[2] - 1, self.dk // self.num_heads), requires_grad=True))
        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)
         
    def forward(self, x):
        batch_size, _, height, width, depth = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hwd = lambda x, channel_depth: torch.reshape(x, (batch_size, self.num_heads, height * width * depth, channel_depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)
        
        # After splitting, shape is [batch_size, num_heads, height, width, dkh or dvh]
        k = self.split_heads_3d(k, self.num_heads)
        q = self.split_heads_3d(q, self.num_heads)
        v = self.split_heads_3d(v, self.num_heads)
        
        # [batch_size, num_heads, height*width*depth, height*width*depth]
        qk = torch.matmul(flatten_hwd(q, dkh), flatten_hwd(k, dkh).transpose(2, 3))
       
        if self.positional_encoding != None:
            raise NotImplementedError
            #qr_h, qr_w, qr_d = self.relative_logits(q)
            #qk += qr_h
            #qk += qr_w
            #qk += qr_d

        weights = F.softmax(qk, dim=-1)
        
        if self.inference:
            self.weights = nn.Parameter(weights)
            
        attn_out = torch.matmul(weights, flatten_hwd(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, self.dv // self.num_heads, height, width, depth))
        attn_out = self.combine_heads_3d(attn_out)
        # Project heads
        attn_out = self.attn_out(attn_out)
        return torch.cat((self.conv_out(x), attn_out), dim=1)

    # Split channels into multiple heads.
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs
    
    def split_heads_3d(self, inputs, num_heads):
        batch_size, channel_depth, height, width, depth = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth, channel_depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs
    
    # Combine heads (inverse of split heads 2d).
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)
    
    def combine_heads_3d(self, inputs):
        batch_size, num_heads, channel_depth, height, width, depth = inputs.size()
        ret_shape = (batch_size, num_heads * channel_depth, height, width, depth)
        return torch.reshape(inputs, ret_shape)
    
    # Compute relative logits for both dimensions.
    def relative_logits(self, q):
        _, num_heads, height, width, depth, dkh = q.size()
        rel_logits_wh = self.relative_logits_1d(q, self.rel_encoding_w, height, width, depth, num_heads,  [0, 1, 2, 4, 3, 5])

        rel_logits_hw = self.relative_logits_1d(torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads,  [0, 1, 4, 2, 5, 3])
        
        #rel_logits_dh = self.relative_logits_1d(q, self.rel_encoding_
        return rel_logits_h, rel_logits_w
    
    # Compute relative logits along one dimenion.
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat((1, num_heads, 1, 1, 1))
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, height, 1, 1))
        # Reshape for adding to the logits.
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing.
    def rel_to_abs(self, x):
        # [batch_size, num_heads*height, L, 2L−1]
        batch_size, num_heads, L, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        col_pad = torch.zeros((batch_size, num_heads, L, 1)).to(self.device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1)).to(self.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements.
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x
