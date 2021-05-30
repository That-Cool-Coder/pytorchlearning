# import libraries
import torch
import torch.nn as nn
from torch.autograd import Variable

import utils

MODEL_FILE_NAME = 'name_classify.pth'

MALE = 0
FEMALE = 1

class NameClassify(nn.Module):
    def __init__(self):
        super(NameClassify, self).__init__()
        self.sigmoid = nn.ReLU()
        self.linear = nn.Linear(utils.MAX_STR_LENGTH, 50)
        self.linear2 = nn.Linear(50, 25)
        self.linear3 = nn.Linear(25, 20)
        self.linear4 = nn.Linear(20, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.sigmoid(x)
        x = self.linear2(x)
        sig = self.sigmoid(x)
        x = self.linear3(x)
        sig = self.sigmoid(x)
        yh = self.linear4(sig)
        return yh