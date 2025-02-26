print('Importing pytorch...')

# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import NumberClassifier, MODEL_FILE_NAME

dev = torch.device('cuda')

print('Initialising model...')
model = NumberClassifier(dev)
print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

# test input
while True:
    user_input = float(input('Enter a number between -1 and 1 '))
    network_input = torch.tensor([user_input],device=dev)
    out = model(network_input)
    result = out[0].cpu().detach().numpy() > .5
    result_text = 'positive' if result else 'negative'
    print('Network says', result_text)