# -*- coding: utf-8 -*-
import torch
from torch import nn

class CFD_ConLSTM(nn.Module):
    def __init__(self, Actfuns,input_shape:list,device):
        super(CFD_ConLSTM, self).__init__()
        self.channels = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.actfun = Actfuns
        self.device = device

        def Convblock(input_channels,output_channels,kernel_size,padding=0):
            layer = nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size,padding=padding),
                nn.BatchNorm2d(output_channels),
                self.actfun
            )
            return layer
        
        self.conv_layers = nn.Sequential(
            Convblock(self.channels, 64, (3, 3), 1),
            Convblock(64, 128, (3, 6), 1),
            Convblock(128, 256, (3, 9), 1),
            nn.MaxPool2d((2,4)),
            Convblock(256, 512, (3, 9)),
            Convblock(512, 512, (3, 15)),
            Convblock(512, 1024, (3, 15)),
            nn.MaxPool2d((2,4)),
        )

        
        self.fc = nn.Linear(self.width, self.height * self.width)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)
        self.lstm = nn.LSTM(input_size=x.shape[1], hidden_size=self.width, batch_first=True).to(self.device)
        x, _ = self.lstm(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = x.reshape(-1,self.channels,self.height,self.width)
        return x
