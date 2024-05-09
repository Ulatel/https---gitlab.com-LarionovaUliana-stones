import torch 
import torch.nn as nn 
from torch import optim 
from torch.optim import lr_scheduler 
import fastai 
from time import time 
import fastcore.all as fc 

def unet_conv(ni, nf, ks=3, stride=1, act=nn.SiLU(), norm=None, bias=True): 
    layers = []
    if norm:
        layers.append(norm(ni))
    if act:
        layers.append(act)
    layers.append(nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2, bias=bias)) 
    return nn.Sequential(*layers)

class UnetResBlock(nn.Module): 
    def __init__(self, ni, nf=None, ks=3, act=nn.SiLU(), norm=nn.BatchNorm2d): 
        super(UnetResBlock, self).__init__() # изменено на super().__init__()
        if nf is None:
            nf = ni
        self.convs = nn.Sequential(unet_conv(ni, nf, ks, act=act, norm=norm), 
                                   unet_conv(nf, nf, ks, act=act, norm=norm)) 
        self.idconv = fc.noop if ni == nf else nn.Conv2d(ni, nf, 1) 

    def forward(self, x):

        return self.convs(x) + self.idconv(x)