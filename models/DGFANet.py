"""LeNet model for DGrPPG."""
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

from pdb import set_trace as st

    

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)


def deconv3x3(in_channels, out_channels, stride=2, padding=1, output_padding=1, bias=False):    
    return nn.ConvTranspose2d(
          in_channels, 
          out_channels,
          kernel_size=3, 
          stride=stride,
          padding=padding,
          output_padding=output_padding,
          bias=bias)


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        # self.conv = conv3x3(in_channels, out_channels)
        self.conv = nn.Sequential(
                    conv3x3(in_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x    


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x



class DOWN(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(DOWN, self).__init__()
        self.mpconv = nn.Sequential(
            Downconv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class DepthEstmator(nn.Module):
    def __init__(self, in_channels=384, out_channels=1):
        super(DepthEstmator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            
            conv3x3(64, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )


    def forward(self, x):
        x = self.conv(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self, in_channels=6):
        super(FeatExtractor, self).__init__()  

        self.inc = inconv(in_channels, 64)

        self.down1 = DOWN(64, 128)
        self.down2 = DOWN(128, 128)
        self.down3 = DOWN(128, 128)


    def forward(self, x):
        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)

        catfeat = torch.cat([re_dx2, re_dx3, dx4],1)

        return catfeat, dx4



class FeatEmbedder(nn.Module):
    def __init__(self, embed_size, in_channels=128):
        super(FeatEmbedder, self).__init__()  

        self.nettype = nettype

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(nn.Linear(512, embed_size),
                                    nn.BatchNorm1d(embed_size),
                                    nn.ReLU(),
                                    nn.Linear(embed_size, 1))
                    

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        feat = x
        pred = self.classifier(x)
         
        return F.normalize(feat, p=2, dim=1), pred 





class Discriminator1(nn.Module):
    def __init__(self, nc=128, ndf=128):
        super(Discriminator1,self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )
        
    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator2(nn.Module):
    def __init__(self, nc=128, ndf=128):
        super(Discriminator2,self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )
        
    def forward(self, x):
        output = self.model(x)
        return output


class Discriminator3(nn.Module):
    def __init__(self, nc=128, ndf=128):
        super(Discriminator3,self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),

        )
        
    def forward(self, x):
        output = self.model(x)
        return output

      
