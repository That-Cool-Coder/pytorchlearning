# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

MODEL_FILE_NAME = 'multiplier.pth'

class Multiplier(nn.Module):
    def __init__(self):
        super(Multiplier, self).__init__()
        self.linear = nn.Linear(2, 10)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(10, 10)
        self.sigmoid2 = nn.Sigmoid()
        self.linear3 = nn.Linear(10, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.sigmoid(x)
        x = self.linear2(x)
        sig = self.sigmoid(x)
        yh = self.linear3(sig)
        return yh