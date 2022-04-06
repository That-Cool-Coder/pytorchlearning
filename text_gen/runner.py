from random import shuffle

print('Importing pytorch...')

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *

print('Initialising model...')
model = TextGen(CHARSET, look_back=LOOK_BACK)
print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

OUT_LEN = 50 

while True:
    user_input = input(f'Enter a prompt (at least {LOOK_BACK} characters): ')
    output = user_input
    for i in range(OUT_LEN):
        output += model.next_char(output)
    print(output)