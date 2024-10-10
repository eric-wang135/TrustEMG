import torch
import torch.nn as nn

class Dense_L_b(nn.Module):
    def __init__(self, in_size, out_size,bias=True):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(in_size, out_size, bias=True),
            nn.BatchNorm1d(out_size),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.dense(x)
        return out

class conv_1d(nn.Module):
    def __init__(self,in_channel,out_channel,frame_size,shift):
        super().__init__()
        self.conv_1d = nn.Sequential(
        nn.Conv1d(in_channel,out_channel,frame_size,shift),
        nn.BatchNorm1d(out_channel),  
        nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.conv_1d(x)
        return out

class CNN_waveform(nn.Module):
    
    def __init__(self,):
        super().__init__()
        K = 8 #kernel size
        H = 64 #initial channel num / original:32
        S = 2 #stride
        self.encoder = nn.Sequential(
            conv_1d(1,H,K,S),
            conv_1d(H,2*H,K,S),
            conv_1d(2*H,4*H,K,S),
            conv_1d(4*H,8*H,K,S),
            nn.Flatten(),
        )
        self.FC = nn.Sequential(
            Dense_L_b(3072,400),
            nn.Dropout(0.5),
            nn.Linear(400,200,bias=True),
        )

    def forward(self,x):
        bs= x.shape[0]
        x = x.reshape(bs,-1,200).unsqueeze(1)
        
        out = self.encoder(x[:,:,0,:])
        out = self.FC(out)
        for i in range(1, 10):
            out = torch.cat([out,self.FC(self.encoder(x[:,:,i,:]))],-1)
        
        return out