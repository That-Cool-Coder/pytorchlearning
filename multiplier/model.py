# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

MODEL_FILE_NAME = 'multiplier.pth'

class Multiplier(nn.Module):
    def __init__(self):
        super(Multiplier, self).__init__()
        self.linear = nn.Linear(2, 50)
        self.activation_function = nn.Sigmoid()
        self.linear2 = nn.Linear(50, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.activation_function(x)
        yh = self.linear2(sig)
        return yh