print('Importing pytorch...')

# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *
import utils

print('Initialising model...')
model = NameClassify()
print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

# test input
while True:
    user_input = input('Enter a name: ')
    network_input = torch.tensor(utils.str_to_list(user_input))
    out = model(network_input)
    result = out.detach().numpy()[0]
    print(f'Network says: {result}')