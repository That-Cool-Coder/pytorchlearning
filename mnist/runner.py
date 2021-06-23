from random import choice

print('Importing libraries..')

# import libraries 
import torch
import torchvision
import matplotlib.pyplot as plt

from model import *
import utils

MODEL_FILE_NAME = 'model.pth'

print('Initialising model...')
model = MNIST()

print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

print('Preparing data...')
data_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('.', train=False, download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
            ])),
    batch_size=1, shuffle=True)

while True:
    all_data = enumerate(data_loader)
    batch_idx, (data, x) = next(all_data)
    with torch.no_grad():
        output = model(data)
    result = output.data.max(1, keepdim=True)[1][0].item()
    plt.imshow(data[0][0], cmap='gray', interpolation='none')
    plt.title(f'Prediction: {result}')
    plt.show()