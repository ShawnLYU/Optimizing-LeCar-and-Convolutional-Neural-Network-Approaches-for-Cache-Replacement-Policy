import copy
import torch
from torch import nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
CACHE_SIZE=100
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, CACHE_SIZE, 2)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=5,            # n_filters
                kernel_size=(5,3),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(  
                in_channels=5,              # input height
                out_channels=10,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=2),    # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        self.out = nn.Sequential(
            nn.Linear(2940,1000),
            nn.Linear(1000,CACHE_SIZE)
            )
        self.linear = nn.Sequential(
            nn.Linear(3,10),
            nn.ReLU(),
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear(500,2)
            )
    def forward(self, x):
        # x = self.conv1(x.unsqueeze(1))
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # output = self.out(x)
        output = self.linear(x)
        return output