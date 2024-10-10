import math, numpy as np
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

def freeze_weight(model): 
    for param in model.parameters():
        param.requires_grad = False

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class conv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift,padding=0,dilation=1):
        super().__init__()
        self.conv_1d = nn.Sequential(
        nn.Conv1d(in_channel,out_channel,frame_size,shift,padding,dilation),
        nn.BatchNorm1d(out_channel),  
        nn.ReLU(inplace=True),
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
        nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        out = self.deconv_1d(x)
        return out

class up(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift,padding=0,out_pad=0):
        super().__init__()
        self.deconv = deconv_1d(in_channel,in_channel//2,frame_size,shift,padding,out_pad)
        self.conv = conv_1d(in_channel,out_channel,frame_size,1,frame_size//2)

    def forward(self, x1, x2):

        x1 = self.deconv(x1)
        diff = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        out = torch.cat([x1,x2],dim=1)
        return self.conv(out)

# U-Net base
class TrustEMGNet_UNetonly(nn.Module):
    def __init__(self,):
        super(TrustEMGNet_UNetonly,self).__init__()
        K = 8 #kernel size
        H = 64 #initial channel num
        S = 2 #stride
        L = 5
        feature_dim = 16*H

        self.down1 = conv_1d(1,H,K,S)
        self.down2 = conv_1d(H,2*H,K,S)
        self.down3 = conv_1d(2*H,4*H,K,S)
        self.down4 = conv_1d(4*H,8*H,K,S)
        self.down5 = conv_1d(8*H,feature_dim,K,S)
  
        self.up0 = up(feature_dim,8*H,K,S)
        self.up1 = up(8*H,4*H,K,S)
        self.up2 = up(4*H,2*H,K,S)
        self.up3 = up(2*H,H,K,S)
        self.up4 = nn.ConvTranspose1d(H,1,K,S)
        
    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        out = self.up0(x5,x4)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out)
        return out[:,:,:emg.shape[1]].squeeze()
        
class TrustEMGNet_DM(TrustEMGNet_UNetonly):
    def __init__(self,):
        super(TrustEMGNet_DM,self).__init__()
        H = 64 #initial channel num
        feature_dim = 16*H
        d_model = feature_dim
        dim_feedforward = d_model*2
        dropout = 0.1
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward = dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bottleneck = x5.permute(0,2,1)
        bottleneck = self.positional_encoding(bottleneck)
        bottleneck = self.transformer_encoder(bottleneck)
        out = self.up0(bottleneck.permute(0,2,1),x4)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out)
        return out[:,:,:emg.shape[1]].squeeze()

class TrustEMGNet_RM(TrustEMGNet_DM):
    # Masking bottleneck
    def __init__(self,):
        super(TrustEMGNet_RM,self).__init__()
        
    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
    
        bottleneck_before_mask = x5.permute(0,2,1)
        mask = self.positional_encoding(bottleneck_before_mask)
        mask = F.sigmoid(self.transformer_encoder(mask))
        
        """
        _, feature_ch = torch.topk(-torch.sum(mask,dim=1), 1024, dim=1) # get the smallest k-value
        for ch in feature_ch:
            #print(ch)
            #print(torch.sum(mask,dim=1)[:,ch])
            mask[:,:,ch] = 1
        """
        
        """
        _, feature_ch = torch.topk(torch.sum(mask,dim=1), 100, dim=1) # get the smallest k-value
        #print(feature_ch.shape)
        for ch in feature_ch:
            #print(ch)
            #print(torch.sum(mask,dim=1)[:,ch])
            mask[:,:,ch] = 0
        """
        
        bottleneck = bottleneck_before_mask * mask #torch.abs(mask-1) #reversemask:

        out = self.up0(bottleneck.permute(0,2,1),x4)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out)
        
        return out[:,:,:emg.shape[1]].squeeze()#, bottleneck_before_mask.permute(0,2,1).squeeze(), mask.permute(0,2,1).squeeze()

class TrustEMGNet_skipall_DM(TrustEMGNet_DM):
    def __init__(self,):
        super(TrustEMGNet_skipall_DM,self).__init__()
        H = 64 #initial channel num
        #skip_num = 4
        feature_dim = [H * (2**(skip_num-1)) for skip_num in range(1,6)]
        self.pe_list = nn.ModuleList()
        self.transformer_list = nn.ModuleList()
        for i in range(len(feature_dim)):
            d_model = feature_dim[i]
            dim_feedforward = d_model*2
            dropout = 0.1
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward = dim_feedforward, dropout=dropout, batch_first=True)
            self.pe_list.append(PositionalEncoding(d_model))
            self.transformer_list.append(nn.TransformerEncoder(encoder_layer, num_layers=1))

    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        f_list, x_list = [],[x1,x2,x3,x4,x5]
        for i in range(5):
            f = x_list[i].permute(0,2,1)
            f = self.pe_list[i](f)
            f = self.transformer_list[i](f).permute(0,2,1)
            f_list.append(f)

        out = self.up0(f_list[-1],f_list[-2])
        out = self.up1(out,f_list[-3])
        out = self.up2(out,f_list[-4])
        out = self.up3(out,f_list[-5])
        out = self.up4(out)
        
        return out[:,:,:emg.shape[1]].squeeze()

class TrustEMGNet_skipall_RM(TrustEMGNet_skipall_DM):
    def __init__(self,):
        super(TrustEMGNet_skipall_RM,self).__init__()

    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        f_list, x_list = [],[x1,x2,x3,x4,x5]

        for i in range(5):
            f = x_list[i].permute(0,2,1)
            mask = self.pe_list[i](f)
            mask = F.sigmoid(self.transformer_list[i](mask))
            f = f * mask
            f = f.permute(0,2,1)
            f_list.append(f)

        out = self.up0(f_list[-1],f_list[-2])
        out = self.up1(out,f_list[-3])
        out = self.up2(out,f_list[-4])
        out = self.up3(out,f_list[-5])
        out = self.up4(out)
        
        return out[:,:,:emg.shape[1]].squeeze()

class TrustEMGNet_LSTM_DM(TrustEMGNet_UNetonly):
    def __init__(self,):
        super(TrustEMGNet_LSTM_DM,self).__init__()
        H = 64 #initial channel num
        feature_dim = 16*H
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, batch_first=True, num_layers=1, dropout=0)

    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bottleneck = x5.permute(0,2,1)
        bottleneck, _ = self.lstm(bottleneck)
        out = self.up0(bottleneck.permute(0,2,1),x4)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out)
        return out[:,:,:emg.shape[1]].squeeze()

class TrustEMGNet_LSTM_RM(TrustEMGNet_LSTM_DM):
    def __init__(self,):
        super(TrustEMGNet_LSTM_RM,self).__init__()
        H = 64 #initial channel num
        feature_dim = 16*H
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=feature_dim, batch_first=True, num_layers=1, dropout=0)

    def forward(self,emg):
        x1 = self.down1(emg.unsqueeze(1))
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        bottleneck = x5.permute(0,2,1)
        mask, _ = self.lstm(bottleneck)
        bottleneck = bottleneck * F.sigmoid(mask)

        out = self.up0(bottleneck.permute(0,2,1),x4)
        out = self.up1(out,x3)
        out = self.up2(out,x2)
        out = self.up3(out,x1)
        out = self.up4(out)
        return out[:,:,:emg.shape[1]].squeeze()