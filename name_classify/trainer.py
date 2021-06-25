import json
from random import shuffle

print('Importing pytorch...')

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import *
import utils

TRAINING_DATA_FILE = 'names.json'

print('Preparing training data...')

file = open(TRAINING_DATA_FILE)
names_str = file.read()
file.close()

names = json.loads(names_str)
shuffle(names)

training_data = names[:len(names)//2]
testing_data = names[len(names)//2:]

Xs = []
Ys = []

for item in training_data:
    Xs.append(utils.str_to_list(item['name']))
    datum = [0, 0]
    if item['gender'] == 'f':
        datum[FEMALE] = 1
    else:
        datum[MALE] = 1
    Ys.append(datum)

Xs = torch.Tensor(Xs)

Ys = torch.Tensor(Ys).reshape(Xs.shape[0], 2)

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
        all_losses.append((current_loss / plot_every).detach().numpy())
        current_loss = 0
    
    # print progress (add 1 to exclude epoch 0 and include final epoch)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1} completed')

print('Saving model...')

torch.save(model.state_dict(), MODEL_FILE_NAME)

print('Creating loss plot...')

# Yes I know importing here isn't ideal
# but it moves this into the 'creating plot' printout
import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.ylabel('Loss')
plt.show()

print('Testing...')

correctness_count = 0
for value in testing_data:
    result, certainty = model.classify_name(utils.str_to_list(value['name']))
    if value['gender'] == 'm':
        correct_result = MALE
    else:
        correct_result = FEMALE
    
    if correct_result == result:
        correctness_count += 1

print(f'Network was correct in {correctness_count / len(testing_data) * 100}% of test cases')