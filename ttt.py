import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict



class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        print(y.shape)
        y=self.conv(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        y=self.sigmoid(y) #bs,1,c
        return x*y.expand_as(x)
    

class SpatiotemporalAttentionModule(nn.Module):
    """Spatiotemporal Attention for SimVP"""

    def __init__(self,h_w):
        super().__init__()
        h, w = h_w
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h=nn.Conv1d(h,h,kernel_size=3,padding=1,bias=False)
        self.conv_w=nn.Conv1d(w,w,kernel_size=3,padding=1,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        u = x.clone()
        x_h = self.pool_h(x).squeeze(-1).permute(0,2,1) 
        x_w = self.pool_w(x).squeeze(-2).permute(0,2,1)
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)
        a_h = a_h.permute(0,2,1).unsqueeze(-1)
        a_w = a_w.permute(0,2,1).unsqueeze(-2)
        a_h = self.sigmoid(x_h) 
        a_w = self.sigmoid(x_w)

        return a_w.expand_as(u) * a_h.expand_as(u) * u




if __name__ == '__main__':
    input=torch.randn(5,10,7,7)
    eca = SpatiotemporalAttentionModule((7,7))
    output=eca(input)
    print(output.shape)