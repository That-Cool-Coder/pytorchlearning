import json
from random import shuffle

import matplotlib.pyplot as plt

print('Importing pytorch...')

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *
import utils

TRAINING_DATA_FILE = 'names_labelled.json'

file = open(TRAINING_DATA_FILE)
training_data_str = file.read()
file.close()

training_data = json.loads(training_data_str)
shuffle(training_data)

Xs = []
Ys = []

for item in training_data:
    Xs.append(utils.str_to_list(item['name']))
    if item['gender'] == 'f':
        Ys.append(FEMALE)
    else:
        Ys.append(MALE)
Xs = torch.Tensor(Xs)

Ys = torch.Tensor(Ys).reshape(Xs.shape[0], 1)

print('Initialising model...')

model = NameClassify()

epochs = 1000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)
all_losses = []
current_loss = 0
plot_every = 1

print('Training...')

for epoch in range(epochs):
    # input training example and return the prediction
    yhat = model.forward(Xs)

    # calculate MSE loss
    loss = mseloss(yhat, Ys)
    
    # backpropogate through the loss gradiants
    loss.backward()

    # update model weights
    optimizer.step()

    # remove current gradients for next iteration
    optimizer.zero_grad()

    # append to loss
    current_loss += loss
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
    
    # print progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch} completed')

print('Saving model...')

torch.save(model.state_dict(), MODEL_FILE_NAME)

print('Done')

plt.plot(all_losses)
plt.ylabel('Loss')
plt.show()