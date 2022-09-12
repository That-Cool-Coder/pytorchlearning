# Really simple (stupidly trivial) network for getting stuff working on GPU
# Due to the tiny network is actually way slower on GPU
# If you add 20000 neurons per layer then GPU is much more performant

print('Importing pytorch...')
# import libraries 
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model import NumberClassifier, MODEL_FILE_NAME

print('Initialising model...')

dev = torch.device('cuda')
torch.set_num_threads(8)

model = NumberClassifier(dev)

print('Creating training data')
# create data
Xs = torch.linspace(-1., 1., 1000).numpy()
np.random.shuffle(Xs)
Xs = torch.tensor(Xs, device=dev).reshape(Xs.shape[0], 1)

Ys = list(map(lambda x: float(x > 0), Xs))
Ys = torch.tensor(Ys, device=dev).reshape(Xs.shape[0], 1)

epochs = 10000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.03)
all_losses = []
current_loss = 0
plot_every = 50

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
    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch + 1} completed')

print('Saving model...')

torch.save(model.state_dict(), MODEL_FILE_NAME)

print('Done')