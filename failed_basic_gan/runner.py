print('Importing libraries...')

import torch

from common import *

print('Initialising model...')
generator = Generator()

print('Loading training...')
generator.load_state_dict(torch.load(GENERATOR_FILE_NAME))

def generate(batch_size=1):
    out = generator(torch.randint(0, 2, size=(batch_size, NUM_LENGTH)).float())
    clean_out = out.detach().numpy().tolist()
    return [list_to_number(item) for item in clean_out]

for i in range(100):
    print(list_to_number(number_to_list(i)))

while True:
    input(f'Network generated: {generate()[0]}')