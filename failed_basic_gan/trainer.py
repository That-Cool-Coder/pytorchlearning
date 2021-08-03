print('Importing libraries...')

import math

import torch
import numpy as np

from common import *

TRAINING_STEPS = 100
BATCH_SIZE = 16
LOG_INTERVAL = 200

def generate_even_data():
    # Generate some random data (training data)

    # Sample BATCH_SIZE number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(MAX_NUM / 2), BATCH_SIZE)

    # create a list of labels all ones because all numbers are even
    labels = [1] * BATCH_SIZE

    # Generate a list of binary numbers for training.
    data = [number_to_list(int(x * 2)) for x in sampled_integers]
    data = [([0] * (NUM_LENGTH - len(x))) + x for x in data]

    return labels, data

print('Initialising models...')

# Models
generator = Generator()
discriminator = Discriminator()

# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# loss
loss = nn.BCELoss()

print('Training...')

for i in range(TRAINING_STEPS):
    # Zero gradients, because apparently that's important
    generator_optimizer.zero_grad()

    # Create noisy input for generator
    noise = torch.randint(0, 2, size=(BATCH_SIZE, NUM_LENGTH)).float()
    generated_data = generator(noise)

    # Generate some actual data
    true_labels, true_data = generate_even_data()
    true_labels = torch.tensor(true_labels).float()
    true_labels.unsqueeze_(1)
    true_data = torch.tensor(true_data).float()

    # Train the generator
    # We invert the labels here and don't train the discriminator because we want the generator
    # to make things the discriminator classifies as true.
    generator_discriminator_out = discriminator(generated_data)
    generator_loss = loss(generator_discriminator_out, true_labels)
    generator_loss.backward()
    generator_optimizer.step()

    # Train the discriminator on the true/generated data
    discriminator_optimizer.zero_grad()
    true_discriminator_out = discriminator(true_data)
    true_discriminator_loss = loss(true_discriminator_out, true_labels)

    # add .detach() here think about this
    generator_discriminator_out = discriminator(generated_data.detach())
    generator_discriminator_loss = loss(generator_discriminator_out,
        torch.zeros(BATCH_SIZE).unsqueeze(1))
    discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
    discriminator_loss.backward()
    discriminator_optimizer.step()

    if (i + 1) % LOG_INTERVAL == 0:
        print(f'Finished training step {i + 1}')

print('Saving generator...')
torch.save(generator.state_dict(), GENERATOR_FILE_NAME)

print('Done')