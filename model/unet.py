import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import ConvBlock, LK_encoder, AttentionBlock, DeConvBlock

class UNet(nn.Module):
    def __init__(self, in_channel=2, n_classes=3, kernel="regular", kernel_dec="regular", lknorm="batchnorm"):
        '''
        kernel: large for lkunet, regular for unet
        '''
        super().__init__()
        self.kernel = kernel
        channel_factor = 8
        if kernel == "regular":
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,64//channel_factor),\
                                          'enc2':ConvBlock(64//channel_factor,128//channel_factor),\
                                          'enc3':ConvBlock(128//channel_factor,256//channel_factor),\
                                          'enc4':ConvBlock(256//channel_factor,512//channel_factor),\
                                          'enc5':ConvBlock(512//channel_factor,1024//channel_factor)}) #,\
                                          #'enc6':ConvBlock(64//channel_factor,80//channel_factor)})
                                         #'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
        elif kernel == "large":
            bias_opt = False
            #self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,64*16//channel_factor, groups=[1, 1]),\
            #                               'enc2':ConvBlock(64*16//channel_factor,128*16//channel_factor),\
            #                               'enc3':LK_encoder(128*16//channel_factor,128*16//channel_factor,norm=lknorm,bias=bias_opt),\
            #                               'enc4':ConvBlock(128*16//channel_factor,256*16//channel_factor),\
            #                               'enc5':LK_encoder(256*16//channel_factor,256*16//channel_factor,norm=lknorm,bias=bias_opt),\
            #                               'enc6':ConvBlock(256*16//channel_factor,512*16//channel_factor),\
            #                               'enc7':LK_encoder(512*16//channel_factor,512*16//channel_factor,norm=lknorm,bias=bias_opt),\
            #                               'enc8':ConvBlock(512*16//channel_factor,1024*16//channel_factor),\
            #                               'enc9':LK_encoder(1024*16//channel_factor,1024*16//channel_factor,norm=lknorm,bias=bias_opt)})
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,64*16//channel_factor, groups=[1, 1]),\
                                           'enc2':ConvBlock(64*16//channel_factor,128*16//channel_factor),\
                                           'enc3':LK_encoder(128*16//channel_factor,128*16//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc4':ConvBlock(128*16//channel_factor,256*16//2//channel_factor),\
                                           'enc5':LK_encoder(256*16//2//channel_factor,256*16//2//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc6':ConvBlock(256*16//2//channel_factor,512*16//4//channel_factor),\
                                           'enc7':LK_encoder(512*16//4//channel_factor,512*16//4//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc8':ConvBlock(512*16//4//channel_factor,1024*16//8//channel_factor),\
                                           'enc9':LK_encoder(1024*16//8//channel_factor,1024*16//8//channel_factor,norm=lknorm,bias=bias_opt)})
        self.kernel_dec = kernel_dec
        if kernel_dec == "regular":
            #self.decoder = nn.ModuleDict({'dec1':ConvBlock(1024*16//channel_factor+512*16//channel_factor,512*16//channel_factor),\
            #                              'dec2':ConvBlock(512*16//channel_factor+256*16//channel_factor,256*16//channel_factor),\
            #                              'dec3':ConvBlock(256*16//channel_factor+128*16//channel_factor,128*16//channel_factor),\
            #                              'dec4':ConvBlock(128*16//channel_factor+64*16//channel_factor,64*16//channel_factor)})
            self.decoder = nn.ModuleDict({'dec1':ConvBlock(1024*16//8//channel_factor+512*16//4//channel_factor,512*16//4//channel_factor),\
                                          'dec2':ConvBlock(512*16//4//channel_factor+256*16//2//channel_factor,256*16//2//channel_factor),\
                                          'dec3':ConvBlock(256*16//2//channel_factor+128*16//channel_factor,128*16//channel_factor),\
                                          'dec4':ConvBlock(128*16//channel_factor+64*16//channel_factor,64*16//channel_factor)})
            #self.decoder = nn.ModuleDict({'dec1':ConvBlock(1024//channel_factor,512//channel_factor),\
            #                              'dec2':ConvBlock(512//channel_factor+512//channel_factor,512//channel_factor),\
            #                              'dec3':ConvBlock(512//channel_factor,256//channel_factor),\
            #                              'dec4':ConvBlock(256//channel_factor+256//channel_factor,256//channel_factor),\
            #                              'dec5':ConvBlock(256//channel_factor,128//channel_factor),\
            #                              'dec6':ConvBlock(128//channel_factor+128//channel_factor,128//channel_factor),\
            #                              'dec7':ConvBlock(128//channel_factor,64//channel_factor),\
            #                              'dec8':ConvBlock(64//channel_factor+64//channel_factor,64//channel_factor)})
        elif kernel_dec == "large":
            bias_opt = False
            self.decoder = nn.ModuleDict({'dec1':ConvBlock(1024//channel_factor+512//channel_factor,512//channel_factor),\
                                          'dec2':LK_encoder(512//channel_factor,512//channel_factor,norm=lknorm,bias=bias_opt),\
                                          'dec3':ConvBlock(512//channel_factor+256//channel_factor,256//channel_factor),\
                                          'dec4':LK_encoder(256//channel_factor,256//channel_factor,norm=lknorm,bias=bias_opt),\
                                          'dec5':ConvBlock(256//channel_factor+128//channel_factor,128//channel_factor),\
                                          'dec6':LK_encoder(128//channel_factor,128//channel_factor,norm=lknorm,bias=bias_opt),\
                                          'dec7':ConvBlock(128//channel_factor+64//channel_factor,64//channel_factor),\
                                          'dec8':LK_encoder(64//channel_factor,64//channel_factor,norm=lknorm,bias=bias_opt)})
            

        #self.conv1 = ConvBlock(64*16//channel_factor,128*16//channel_factor)
        #self.conv2 = nn.Sequential(nn.Conv3d(128*16//channel_factor,64*16//channel_factor,1,bias=False),nn.InstanceNorm3d(64*16//channel_factor),nn.ReLU(inplace=True),\
        self.conv1 = ConvBlock(64*16//channel_factor,128*16//2//channel_factor)
        self.conv2 = nn.Sequential(nn.Conv3d(128*16//2//channel_factor,64*16//channel_factor,1,bias=False),nn.InstanceNorm3d(64*16//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(64*16//channel_factor,64*16//channel_factor,1,bias=False),nn.InstanceNorm3d(64*16//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(64*16//channel_factor,n_classes,1))
        
    def forward(self, x):
        y = []
        upsample = nn.Upsample(scale_factor=2,mode='nearest')#mode='trilinear')
        # encoder
        if self.kernel == "regular":
            for i in range(5):
                x = self.encoder['enc'+str(i+1)](x)
                if(i<4):
                    y.append(x)
                    #x = self.encoder['down'+str(i+1)](x)
                    x = F.max_pool3d(x,2)
        elif self.kernel == "large":
            x = self.encoder['enc1'](x)
            y.append(x)
            #x = self.encoder['down1'](x)
            x = F.max_pool3d(x,2)
            for i in range(1,9):
                x = self.encoder['enc'+str(i+1)](x)
                if i % 2 == 0 and i < 7:
                    y.append(x)
                    #x = self.encoder['down'+str(i//2+1)](x)
                    x = F.max_pool3d(x,2)
        
        # decoder
        if self.kernel_dec == "regular":
            for i in range(4):
                #x = torch.cat((self.decoder['up'+str(i+1)](x),y.pop()),1)
                x = torch.cat((upsample(x),y.pop()),1)
                x = self.decoder['dec'+str(i+1)](x)
            #for i in range(4):
            #    #x = torch.cat((self.decoder['up'+str(i+1)](x),y.pop()),1)
            #    x = self.decoder['dec'+str(i*2+1)](x)
            #    x = torch.cat((upsample(x),y.pop()),1)
            #    x = self.decoder['dec'+str(i*2+2)](x)
        elif self.kernel_dec == "large":
            for i in range(8):
                if i % 2 == 0:
                    #x = torch.cat((self.decoder['up'+str(i+1)](x),y.pop()),1)
                    x = torch.cat((upsample(x),y.pop()),1)
                x = self.decoder['dec'+str(i+1)](x)
        x = self.conv1(x)
        return self.conv2(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channel=2, n_classes=3, kernel="regular", kernel_dec="regular", lknorm="batchnorm"):
        '''
        kernel: large for lkunet, regular for unet
        '''
        super().__init__()
        self.kernel = kernel
        channel_factor = 1
        if kernel == "regular":
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,32//channel_factor),\
                                          'enc2':ConvBlock(32//channel_factor,64//channel_factor),\
                                          'enc3':ConvBlock(64//channel_factor,128//channel_factor),\
                                          'enc4':ConvBlock(128//channel_factor,256//channel_factor),\
                                          'enc5':ConvBlock(256//channel_factor,512//channel_factor)}) #,\
                                          #'enc6':ConvBlock(64//channel_factor,80//channel_factor)})
                                         #'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
        elif kernel == "large":
            bias_opt = False
            self.encoder = nn.ModuleDict({'enc1':ConvBlock(in_channel,32//channel_factor, groups=[1, 1]),\
                                           'enc2':ConvBlock(32//channel_factor,64//channel_factor),\
                                           'enc3':LK_encoder(64//channel_factor,64//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc4':ConvBlock(64//channel_factor,128//channel_factor),\
                                           'enc5':LK_encoder(128//channel_factor,128//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc6':ConvBlock(128//channel_factor,256//channel_factor),\
                                           'enc7':LK_encoder(256//channel_factor,256//channel_factor,norm=lknorm,bias=bias_opt),\
                                           'enc8':ConvBlock(256//channel_factor,512//channel_factor),\
                                           'enc9':LK_encoder(512//channel_factor,512//channel_factor,norm=lknorm,bias=bias_opt)})
        self.kernel_dec = kernel_dec
        if kernel_dec == "regular":
            #self.decoder = nn.ModuleDict({'dec1':ConvBlock(80//channel_factor+64//channel_factor,64//channel_factor),\
            #                              'dec2':ConvBlock(64//channel_factor+48//channel_factor,48//channel_factor),\
            #                              'dec3':ConvBlock(48//channel_factor+48//channel_factor,48//channel_factor),\
            #                              'dec4':ConvBlock(48//channel_factor+32//channel_factor,32//channel_factor)})
            self.decoder = nn.ModuleDict({'att1':AttentionBlock(512//channel_factor,256//channel_factor,256//channel_factor),\
                                          'dec1':ConvBlock(512//channel_factor+256//channel_factor,256//channel_factor),\
                                          'att2':AttentionBlock(256//channel_factor,128//channel_factor,128//channel_factor),\
                                          'dec2':ConvBlock(256//channel_factor+128//channel_factor,128//channel_factor),\
                                          'att3':AttentionBlock(128//channel_factor,64//channel_factor,64//channel_factor),\
                                          'dec3':ConvBlock(128//channel_factor+64//channel_factor,64//channel_factor),\
                                          'att4':AttentionBlock(64//channel_factor,32//channel_factor,32//channel_factor),\
                                          'dec4':ConvBlock(64//channel_factor+32//channel_factor,32//channel_factor)})
        elif kernel_dec == "large":
            bias_opt = False
            raise NotImplementedError
        self.conv1 = ConvBlock(32//channel_factor,64//channel_factor)
        self.conv2 = nn.Sequential(nn.Conv3d(64//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,32//channel_factor,1,bias=False),nn.InstanceNorm3d(32//channel_factor),nn.ReLU(inplace=True),\
                                 nn.Conv3d(32//channel_factor,n_classes,1))
    def forward(self, x):
        y = []
        upsample = nn.Upsample(scale_factor=2,mode='nearest') #'trilinear')
        if self.kernel == "regular":
            for i in range(4+1):
                x = self.encoder['enc'+str(i+1)](x)
                if(i<3+1):
                    y.append(x)
                    x = F.max_pool3d(x,2)
        elif self.kernel == "large":
            x = self.encoder['enc1'](x)
            y.append(x)
            x = F.max_pool3d(x,2)
            for i in range(1,9):
                x = self.encoder['enc'+str(i+1)](x)
                if i % 2 == 0 and i < 7:
                    y.append(x)
                    x = F.max_pool3d(x,2)

        if self.kernel_dec == "regular":
            for i in range(3+1):
                # upsample + attention + conv
                x = upsample(x)
                x_a = self.decoder['att'+str(i+1)](g=x,x=y.pop())

                x = torch.cat((x,x_a),1)
                x = self.decoder['dec'+str(i+1)](x)
        elif self.kernel_dec == "large":
            raise NotImplementedError
        x = self.conv1(x)
#         return upsample(self.conv2(x))
        return self.conv2(x)

class UNetReg(nn.Module):
    def __init__(
        self, 
        in_channel=2, 
        n_classes=3, 
        kernel='regular', 
        kernel_dec='regular', 
        lknorm='batchnorm', 
        model_type='regular'
    ):
        super().__init__()
        if model_type == "regular":
            self.unet = UNet(in_channel=in_channel,
                             n_classes=n_classes,
                             kernel=kernel, 
                             kernel_dec=kernel_dec,
                             lknorm=lknorm)
        elif model_type == "attention":
            self.unet = AttentionUNet(in_channel=in_channel, 
                                      n_classes=n_classes, 
                                      kernel=kernel,
                                      kernel_dec=kernel_dec,
                                      lknorm=lknorm)



    def forward(self, x, y):
        input = torch.cat((x.to(torch.float), y.to(torch.float)), 1)
        output = self.unet(input)
        return output
