# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

from xor_model import XOR

MODEL_FILE_NAME = 'xor.pth'

model = XOR()
model.load_state_dict(torch.load(MODEL_FILE_NAME))

# test input
while True:
    user_input = input('Enter 2 values seperated by space: ').split(' ')
    if len(user_input) != 2:
        print('Invalid input')
        continue
    network_input = torch.tensor([float(user_input[0]), float(user_input[1])])
    out = model(network_input)
    print('Network says:', int(out.round().detach().numpy()[0]))