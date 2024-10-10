import torch
import torch.nn as nn

class Dense_L(nn.Module):

    def __init__(self, in_size, out_size,bias=True):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.dense(x)
        return out

class conv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift,padding=0,dilation=1):
        super().__init__()
        self.conv_1d = nn.Sequential(
        nn.Conv1d(in_channel,out_channel,frame_size,shift,padding,dilation),
        nn.BatchNorm1d(out_channel),  
        nn.ReLU(),
        )
    def forward(self, x):
        out = self.conv_1d(x)
        return out

class deconv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift,padding=0,out_pad=0,dilation=1):
        super().__init__()
        self.deconv_1d = nn.Sequential(
        nn.ConvTranspose1d(in_channel,out_channel,frame_size,shift,padding,output_padding=out_pad,dilation=dilation),
        nn.BatchNorm1d(out_channel),  
        nn.ReLU(),
        )
    def forward(self, x):
        out = self.deconv_1d(x)
        return out

class FCN(nn.Module):
    def __init__(self,):
        super().__init__()
        K = 8 #kernel size
        H = 64 #initial channel num
        S = 2 #stride
        L = 5
        feature_dim = 16*H
        self.encoder = nn.Sequential(
            conv_1d(1,H,K,S),
            conv_1d(H,2*H,K,S),
            conv_1d(2*H,4*H,K,S),
            conv_1d(4*H,8*H,K,S),
            conv_1d(8*H,feature_dim,K,S)
        )
        self.decoder = nn.Sequential(
            deconv_1d(feature_dim,8*H,K,S,out_pad=1),
            deconv_1d(8*H,4*H,K,S,out_pad=1),
            deconv_1d(4*H,2*H,K,S),
            deconv_1d(2*H,H,K,S),
            nn.ConvTranspose1d(H,1,K,S)
        )
    def forward(self,emg):
        f = self.encoder(emg.unsqueeze(1))
        out = self.decoder(f)
        return out[:,:,:emg.shape[1]].squeeze()
        


