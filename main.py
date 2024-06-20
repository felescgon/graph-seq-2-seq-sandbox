import numpy as np
from Discriminator import Discriminator
from Generator import Generator
import torch

n_features = 4
sequence_length = 6
batch_size = 16
input = torch.as_tensor(np.ones((batch_size, sequence_length, n_features))).float()
generator = Generator(n_features, n_features*2, sequence_length)
res = generator(input)
discriminator = Discriminator(n_features, n_features*2, 1)
res2 = discriminator(res)
