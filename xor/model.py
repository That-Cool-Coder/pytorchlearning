# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

MODEL_FILE_NAME = 'xor.pth'

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.Sigmoid(x)
        yh = self.linear2(sig)
        return yh