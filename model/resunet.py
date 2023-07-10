import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
   
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

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.layer4 = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        self.resnet.maxpool = nn.MaxPool3d(2)
        self.resnet.fc = nn.Sequential(
                            nn.Unflatten(1, (8*32//2, 28, 24, 28)),
                            nn.Upsample(scale_factor=2, mode="trilinear")
                         )

        self.resnet.conv1 = nn.Conv2d(2, 64, 5, stride=1, padding=2)
        self.resnet.layer2[0].conv1.stride = (1, 1)
        self.resnet.layer2[0].downsample[0].stride = 1

        count = 0; count2 = 0
        for name, module in self.resnet.named_modules():
            if isinstance(module, nn.Conv2d):
                before = get_layer(self.resnet, name)
                after = nn.Conv3d(before.in_channels//2,
                                  before.out_channels//2,
                                  int(torch.tensor(before.kernel_size)[0]),
                                  stride=int(torch.tensor(before.stride).view(-1)[0]),
                                  padding=before.padding[0])
                set_layer(self.resnet, name, after); count += 1
        
            if isinstance(module, nn.BatchNorm2d):
                before = get_layer(self.resnet, name)
                after = nn.BatchNorm3d(before.num_features//2)
                set_layer(self.resnet, name, after); count2 += 1
        print(count,'# Conv2d > Conv3d','and',count2,'#BatchNorms')


    def forward(self, x):
        return self.resnet(x)

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer

def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({'enc1':ConvBlock(256,32),\
                                      'enc2':ConvBlock(32,48),\
                                      'enc3':ConvBlock(48,48),\
                                      'enc4':ConvBlock(48,64)})
        self.decoder = nn.ModuleDict({'dec1':ConvBlock(64+48,48),\
                                      'dec2':ConvBlock(48+48,48),\
                                      'dec3':ConvBlock(48+32,32)})
        self.conv1 = ConvBlock(32,64)
        self.conv2 = nn.Sequential(nn.Conv3d(64,32,1,bias=False),nn.InstanceNorm3d(32),nn.ReLU(inplace=True),\
                                   nn.Conv3d(32,32,1,bias=False),nn.InstanceNorm3d(32),nn.ReLU(inplace=True),\
                                   nn.Conv3d(32,3,1))

    def forward(self, x):
        y = []
        upsample = nn.Upsample(scale_factor=2,mode='trilinear')
        for i in range(4):
            #print('input:{}'.format(x.shape))
            #print(self.encoder['enc'+str(i+1)])
            x = self.encoder['enc'+str(i+1)](x)
            #print('output:{}'.format(x.shape))
            if(i < 3):
                y.append(x)
                #print('input:{}'.format(x.shape))
                #print(F.max_pool3d)
                x = F.max_pool3d(x,2) 
                #print('output:{}'.format(x.shape))
        for i in range(3):
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
        return upsample(self.conv2(x))

class ResUNetReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()
        self.unet = UNet()

    def forward(self, x, y):
        input = torch.cat((self.resnet(x.to(torch.float)), self.resnet(y.to(torch.float))), 1)
        output = self.unet(input)
        return output
