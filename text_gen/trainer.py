import json
from random import shuffle

print('Importing pytorch...')

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *

print('Initialising model...')

model = TextGen(CHARSET, LOOK_BACK)

print('Preparing training data...')

with open('lipsum.txt', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\n', ' ')
content = content.lower()
content = ''.join([c if c in CHARSET else '' for c in content])

inputs = []
outputs = []

for i in range(LOOK_BACK, len(content)):
    inputs.append(model.text_to_tensor(content[i - LOOK_BACK:i]))
    outputs.append(model.char_to_tensor(content[i]))

inputs = torch.Tensor(inputs)
outputs = torch.Tensor(outputs).reshape(inputs.shape[0], len(CHARSET))

print('Training...')

epochs = 1000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)

for epoch in range(epochs):
    # input training example and return the prediction
    yhat = model.forward(inputs)

    # calculate MSE loss
    loss = mseloss(yhat, outputs)
    
    # backpropogate through the loss gradiants
    loss.backward()

    # update model weights
    optimizer.step()

    # remove current gradients for next iteration
    optimizer.zero_grad()
    
    # print progress (add 1 to exclude epoch 0 and include final epoch)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1} completed')

print('Saving model...')

torch.save(model.state_dict(), MODEL_FILE_NAME)