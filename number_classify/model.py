# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

MODEL_FILE_NAME = 'num_classify.pth'

class NumberClassifier(nn.Module):
    # accepts number between -1 and 1 and decides if it is positive or negative
    # extremely simple thing to try and get everything working

    def __init__(self):
        super(NumberClassifier, self).__init__()
        self.linear = nn.Linear(1, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.Sigmoid(x)
        yh = self.linear2(sig)
        return yh