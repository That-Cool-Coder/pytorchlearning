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
        self.activation_function = nn.Sigmoid()
        self.linear = nn.Linear(utils.MAX_STR_LENGTH * len(utils.CHARSET), 100)
        self.linear2 = nn.Linear(100, 25)
        self.linear3 = nn.Linear(25, 1)

    def forward(self, input):
        x = self.linear(input)
        sig = self.activation_function(x)
        x = self.linear2(x)
        sig = self.activation_function(x)
        yh = self.linear3(sig)
        return yh