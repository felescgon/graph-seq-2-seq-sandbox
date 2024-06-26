import numpy as np
from Discriminator import Discriminator
from Generator import Generator
import torch
import torch.nn as nn

n_features = 4
#tiene que ser múltiplo de 5
sequence_length = 13
batch_size = 16
z_dimension_expansor = 4
z_dimension = sequence_length * n_features * z_dimension_expansor
generator_feature_maps_dimension = 64
discriminator_feature_maps_dimension = 64
nz= 100


input = torch.as_tensor(np.ones((batch_size, sequence_length, n_features))).float()
# generator = Generator(n_features, n_features*2, sequence_length)
# res = generator(input)
# discriminator = Discriminator(n_features, n_features*2, 1)
# res2 = discriminator(res)

#noise = torch.randn(b_size, nz, 1, 1, device=device)
generator = Generator(sequence_length=sequence_length, n_features = n_features)
noise = torch.randn(batch_size, nz, 1)
res = generator(noise)
print(res.shape)