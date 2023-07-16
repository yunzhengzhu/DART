import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
                nn.Conv3d(in_channels,out_channels,3,padding=1,bias=False),\
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
                nn.Conv3d(out_channels,out_channels,1,bias=False),\
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

class LK_encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        padding=2,
        bias=False,
        norm='batchnorm',
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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
        oneKernel = self.layer_oneKernel(inputs)
        # if self.layer_indentity:
        outputs = regularKernel + largeKernel + oneKernel + inputs
        # else:
        # outputs = regularKernel + largeKernel + oneKernel
        # if self.batchnorm:
        # outputs = self.layer_batchnorm(self.layer_batchnorm)
        return self.layer_nonlinearity(outputs)

class UNet(nn.Module):
    def __init__(self, in_channel=2, n_classes=3, kernel='regular', lknorm='batchnorm'):
        '''
        kernel: large for lkunet, regular for unet
        '''
        super().__init__()
        self.kernel = kernel
        channel_factor = 1
        if kernel == 'regular':
            #self.encoder = nn.ModuleDict({'enc1':ConvBlock(2,32//channel_factor),\
            #                              'enc2':ConvBlock(32//channel_factor,48//channel_factor),\
            #                              'enc3':ConvBlock(48//channel_factor,48//channel_factor),\
            #                              'enc4':ConvBlock(48//channel_factor,64//channel_factor),\
            #                              'enc5':ConvBlock(64//channel_factor,80//channel_factor)}) #,\
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,32//channel_factor),\
                                          'enc2':ConvBlock(32//channel_factor,64//channel_factor),\
                                          'enc3':ConvBlock(64//channel_factor,128//channel_factor),\
                                          'enc4':ConvBlock(128//channel_factor,256//channel_factor),\
                                          'enc5':ConvBlock(256//channel_factor,512//channel_factor)}) #,\
                                          #'enc6':ConvBlock(64//channel_factor,80//channel_factor)})
                                         #'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
        elif kernel == 'large':
            bias_opt = False
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,32//channel_factor),\
                                           'enc2':ConvBlock(32//channel_factor,64//channel_factor),\
                                           'enc3':LK_encoder(64//channel_factor,64//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc4':ConvBlock(64//channel_factor,128//channel_factor),\
                                           'enc5':LK_encoder(128//channel_factor,128//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc6':ConvBlock(128//channel_factor,256//channel_factor),\
                                           'enc7':LK_encoder(256//channel_factor,256//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc8':ConvBlock(256//channel_factor,512//channel_factor),\
                                           'enc9':LK_encoder(512//channel_factor,512//channel_factor,norm=lknorm,bias=bias_opt)})

        #self.decoder = nn.ModuleDict({'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
        #                              'dec2':ConvBlock(64//channel_factor+48//channel_factor,48//channel_factor),\
        #                              'dec3':ConvBlock(48//channel_factor+48//channel_factor,48//channel_factor),\
        #                              'dec4':ConvBlock(48//channel_factor+32//channel_factor,32//channel_factor)})
        self.decoder = nn.ModuleDict({'dec1':ConvBlock(512//channel_factor+256//channel_factor,256//channel_factor),\
                                      'dec2':ConvBlock(256//channel_factor+128//channel_factor,128//channel_factor),\
                                      'dec3':ConvBlock(128//channel_factor+64//channel_factor,64//channel_factor),\
                                      'dec4':ConvBlock(64//channel_factor+32//channel_factor,32//channel_factor)})
        self.conv1 = ConvBlock(32//channel_factor,64//channel_factor)
        self.conv2 = nn.Sequential(nn.Conv3d(64//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,n_classes,1))
    def forward(self, x):
        y = []
        upsample = nn.Upsample(scale_factor=2,mode='trilinear')
        if self.kernel == "regular":
            for i in range(4+1):
                #print('input:{}'.format(x.shape))
                #print(self.encoder['enc'+str(i+1)])
                x = self.encoder['enc'+str(i+1)](x)
                #print('output:{}'.format(x.shape))
                if(i<3+1):
                    y.append(x)
                    #print('input:{}'.format(x.shape))
                    #print(F.max_pool3d)
                    x = F.max_pool3d(x,2)
                    #print('output:{}'.format(x.shape))
        elif self.kernel == "large":
            x = self.encoder['enc1'](x)
            y.append(x)
            x = F.max_pool3d(x,2)
            for i in range(1,9):
                x = self.encoder['enc'+str(i+1)](x)
                if i % 2 == 0 and i < 7:
                    y.append(x)
                    x = F.max_pool3d(x,2)

        for i in range(3+1):
            #if(i<3):
            #print('input:{}'.format(x.shape))
            x = torch.cat((upsample(x),y.pop()),1)
            #print('output:{}'.format(x.shape))
            #print('input:{}'.format(x.shape))
            #print(self.decoder['dec'+str(i+1)])
            x = self.decoder['dec'+str(i+1)](x)
            #print('output:{}'.format(x.shape))
        #print('input:{}'.format(x.shape))
        #print(self.conv1)
        x = self.conv1(x)
        #print('output:{}'.format(x.shape))
#         return upsample(self.conv2(x))
        return self.conv2(x)


class UNetReg(nn.Module):
    def __init__(self, in_channel=2, n_classes=3, kernel='regular', lknorm='batchnorm'):
        super().__init__()
        self.unet = UNet(in_channel=in_channel,
                         n_classes=n_classes,
                         kernel=kernel, 
                         lknorm=lknorm)

    def forward(self, x, y):
        input = torch.cat((x.to(torch.float), y.to(torch.float)), 1)
        output = self.unet(input)
        return output
