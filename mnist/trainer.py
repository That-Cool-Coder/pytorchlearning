import json
from random import shuffle

print('Importing libraries...')

import torch
import torch.optim as optim
import torchvision

from model import *
import utils

# Random constants
MODEL_FILE_NAME = 'model.pth'
NUM_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL= 50

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

print('Preparing training data...')
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=True, download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ])),
    batch_size=BATCH_SIZE_TRAIN, shuffle=True)

print('Preparing testing data...')
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=False, download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ])),
    batch_size=BATCH_SIZE_TRAIN, shuffle=True)

print('Initialising model...')

model = MNIST()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
    momentum=MOMENTUM)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(NUM_EPOCHS + 1)]

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL== 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), MODEL_FILE_NAME)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

print('Training...')

test()
for epoch in range(1, NUM_EPOCHS + 1):
    train(epoch)
    test()

print('Saving model...')

torch.save(model.state_dict(), MODEL_FILE_NAME)

print('Creating loss plot...')

# Yes I know importing here isn't ideal
# but it moves this into the 'creating plot' printout
import matplotlib.pyplot as plt

plt.plot(train_counter, train_losses)
plt.ylabel('Loss')
plt.ylabel('Training duration')
plt.show()