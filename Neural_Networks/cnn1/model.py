import copy
import torch
from torch import nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, CACHE_SIZE, 2)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8,            # n_filters
                kernel_size=(30,3),              # filter size
                stride=1,                   # filter movement/step
                padding=0,                  # if want same width and length of this image afte#
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=8,              # input height
                out_channels=16,            # n_filters
                kernel_size=(30,1),              # filter size
                stride=1,                   # filter movement/step
                padding=2),    # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        self.out = nn.Sequential(
            nn.Linear(3680,100),
            )
    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 *#
        output = self.out(x)
        return output

