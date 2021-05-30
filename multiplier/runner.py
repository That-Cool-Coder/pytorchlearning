print('Importing pytorch...')

# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Multiplier, MODEL_FILE_NAME

print('Initialising model...')
model = Multiplier()
print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

# test input
while True:
    user_input = input('Enter 2 values seperated by space: ').split(' ')
    if len(user_input) != 2 or '' in user_input:
        print('Invalid input')
        continue
    network_input = torch.tensor([float(user_input[0]), float(user_input[1])])
    out = model(network_input)
    result = out.detach().numpy()[0]
    print(f'''Network says: {round(result)}''')