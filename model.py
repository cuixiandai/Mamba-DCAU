import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

ic=200
oc=256
ws=13
fs=(ws+1)//2
png_in=fs*fs

tc=oc
L1O=128

num_class=16

d_model=ws*ws

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x):
        return self.conv(x)

def BasicConv(in_channels, out_channels, kernel_size, stride=1, padding=None):
    if not padding:
        padding = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        padding = padding
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),)

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in=ic,num_encoder_layers=3,dim_feedforward=384,nhead=8,dropout=0.1):
        super(EncoderLayers, self).__init__()
        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_norm =LayerNorm(encoder_in)#d_model
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x    

class MyModel(torch.nn.Module):
    def __init__(self,batch_size=32,bilinear=True):
        super(MyModel, self).__init__()
        
        self.conv0 = BasicConv(in_channels=ic, out_channels=oc, kernel_size=3, stride=2, padding=1) 
        #Block
        self.conv1 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        
        self.inc= DoubleConv(ic,oc)
        self.down1= Down(oc,512)
        self.down2=Down(512,1024)

        self.up2= Up(512,oc,bilinear)
        self.up1= Up(1024,512,bilinear)
        
        self.outc= OutConv(oc,num_class)
        #MLP
        self.dropout=Dropout(0.5)
        self.linear1=Linear(oc*ws*ws, L1O)
        self.linear2=Linear(L1O, num_class)   

        self.mamba = MambaLayers(d_model=ic,n_layers=1)
      
        self.encoder1 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8)
        self.encoder2 = EncoderLayers(encoder_in=ic,num_encoder_layers=1,dim_feedforward=512,nhead=8) 
        
        self.position_embedding = position_embeddings(torch.arange(batch_size*ic*ws*ws), 1)  
        self.BN=nn.BatchNorm2d(ic)
        self.lmd = nn.Parameter(torch.tensor(0.5))
    def forward(self, x):
        batch_size=x.shape[0]     
        
        x=self.mamba(x) 
        
        x=self.BN(x)
        
        position_ids = torch.arange(batch_size*ic*ws*ws, dtype=torch.long, device=x.device)  
        position_embeds = self.position_embedding(position_ids).reshape(batch_size, ic,ws,ws) 
        x=x+position_embeds
        
        x1=self.encoder1(x)
        
        x=x.transpose(2,3)
        x=self.encoder2(x)
        x=x.transpose(2,3) 
        
        x=(1-self.lmd)*x+self.lmd*x1
        
        x1= self.inc(x)
        
        x2= self.down1(x1)

        x3= self.down2(x2)
      
        x= self.up1(x3,x2)

        x=self.up2(x,x1)
        x=self.dropout(x)
        x=self.outc(x)

        _, _, h, w= x.shape
        
        x = x[:, :, (h - 1) // 2, (w - 1) // 2]

        return x