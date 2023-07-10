import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,out_channels,3,padding=1,bias=False),\
                                   nn.InstanceNorm3d(out_channels),nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels,out_channels,1,bias=False),\
                                   nn.InstanceNorm3d(out_channels),nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

class UNet(nn.Module):
    def __init__(self, kernel='regular'):
        '''
        kernel: large for lkunet, regular for unet
        '''
        super().__init__()
        self.kernel = kernel
        channel_factor = 1
        if kernel == 'regular':
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(2,32//channel_factor),\
                                          'enc2':ConvBlock(32//channel_factor,48//channel_factor),\
                                          'enc3':ConvBlock(48//channel_factor,48//channel_factor),\
                                          'enc4':ConvBlock(48//channel_factor,64//channel_factor),\
                                          'enc5':ConvBlock(64//channel_factor,80//channel_factor)}) #,\
                                          #'enc6':ConvBlock(64//channel_factor,80//channel_factor)})
                                         #'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
        elif kernel == 'large':
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(2,32//channel_factor),\
                                           'enc2':ConvBlock(32//channel_factor,48//channel_factor),\
                                           'enc3':LK_encoder(48//channel_factor,48//channel_factor,batchnorm=True,bias=True),\
                                           'enc4':ConvBlock(48//channel_factor,48//channel_factor),\
                                           'enc5':LK_encoder(48//channel_factor,48//channel_factor,batchnorm=True,bias=True),\
                                           'enc6':ConvBlock(48//channel_factor,64//channel_factor),\
                                           'enc7':LK_encoder(64//channel_factor,64//channel_factor,batchnorm=True,bias=True),\
                                           'enc8':ConvBlock(64//channel_factor,80//channel_factor),\
                                           'enc9':LK_encoder(80//channel_factor,80//channel_factor,batchnorm=True,bias=True)})

        self.decoder = nn.ModuleDict({'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
                                      'dec2':ConvBlock(64//channel_factor+48//channel_factor,48//channel_factor),\
                                      'dec3':ConvBlock(48//channel_factor+48//channel_factor,48//channel_factor),\
                                      'dec4':ConvBlock(48//channel_factor+32//channel_factor,32//channel_factor)})
        self.conv1 = ConvBlock(32//channel_factor,64//channel_factor)
        self.conv2 = nn.Sequential(nn.Conv3d(64//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,3,1))
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
    def __init__(self, kernel='regular'):
        super().__init__()
        self.unet = UNet(kernel=kernel)

    def forward(self, x, y):
        input = torch.cat((x.to(torch.float), y.to(torch.float)), 1)
        output = self.unet(input)
        return output
