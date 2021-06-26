import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LENGTH = 8
MAX_NUM = 2 ** NUM_LENGTH
GENERATOR_FILE_NAME = 'generator.pth'

def number_to_list(number:int):
    if number > MAX_NUM or number < 0:
        raise ValueError(f'Invalid value for number_to_list: {number}')
    return [1 if digit=='1' else 0 for digit in format(number, '08b')]

def list_to_number(list_to_convert):
    result = 0
    for idx in range(len(list_to_convert)):
        result += 2 ** (len(list_to_convert) - idx - 1) * round(list_to_convert[idx])
    return result

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(NUM_LENGTH, NUM_LENGTH)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense_layer = nn.Linear(NUM_LENGTH, 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))