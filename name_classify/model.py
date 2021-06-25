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
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(utils.MAX_STR_LENGTH * len(utils.CHARSET), 100)
        self.linear2 = nn.Linear(100, 15)
        self.linear3 = nn.Linear(15, 2)

    def forward(self, input):
        x = self.linear(input)
        val = self.relu(x)
        x = self.linear2(val)
        val = self.sigmoid(x)
        yh = self.linear3(val)
        return yh
    
    def classify_name(self, name:str):
        network_input = torch.tensor(utils.str_to_list(name))
        output = self(network_input).detach().numpy().tolist()
        gender = output.index(max(output))
        certainty = abs(output[0] - output[1])

        return (gender, certainty)