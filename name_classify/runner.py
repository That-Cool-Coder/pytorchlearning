from random import shuffle

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
    result, certainty = model.classify_name(user_input)

    if result == MALE:
        gender = 'male'
    else:
        gender = 'female'

    print(f'Network says: {gender} (certainty: {certainty})')