import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from data_load import get_alibaba_encoded_machine_usage
from Discriminator import Discriminator
from Generator import Generator
from helpers import index_splitter, make_balanced_sampler
from trainer import StepByStep


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

seq_len = 64 # No se puede cambiar sin cambiar la estructura convolucional del generador
batch_size = 128
x, y, autoencoder = get_alibaba_encoded_machine_usage(5120, seq_len)
train_idx, val_idx = index_splitter(len(x), [80,20])

x_tensor = torch.as_tensor(x)
y_tensor = torch.as_tensor(y)

x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]

#scale x
reshaped_x_train_tensor = x_train_tensor.reshape(-1,1)

scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(reshaped_x_train_tensor)

scaled_x_train_tensor = torch.as_tensor(scaler.transform(reshaped_x_train_tensor).reshape(x_train_tensor.shape))

x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

reshaped_x_val_tensor = x_val_tensor.reshape(-1,1)
scaled_x_val_tensor = torch.as_tensor(scaler.transform(reshaped_x_val_tensor).reshape(x_val_tensor.shape))


#making the samples
#sampler = make_balanced_sampler(y_train_tensor)

train_dataset = TensorDataset(scaled_x_train_tensor.float(), y_train_tensor.view(-1, 1).float())
test_dataset = TensorDataset(scaled_x_val_tensor.float(), y_val_tensor.view(-1, 1).float())

#train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
#The examples should be uniformly distributed among classes
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

num_layers = 2
rnn_layer = nn.LSTM
bidirectional = False
n_discriminator_outputs = 1
n_features = x_train_tensor.shape[2]
hidden_dim_discriminator = 16

batch_norm = True
dropout = 0.2

torch.manual_seed(21)
generator = Generator(sequence_length=seq_len, n_features = n_features)
generator.apply(weights_init)
discriminator = Discriminator(n_features=n_features, hidden_dim=hidden_dim_discriminator, n_outputs=n_discriminator_outputs, num_layers=num_layers, bidirectional=bidirectional, rnn_layer=rnn_layer, batch_norm=batch_norm, dropout=dropout)
loss = nn.BCELoss()

generator_lr = 0.0001
discriminator_lr = 0.0001

generator_optimizer = optim.Adam(generator.parameters(), lr=generator_lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_lr)

sbs_rnn = StepByStep(generator, discriminator, loss, generator_optimizer, discriminator_optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
print("antes de entrenamiento")
sbs_rnn.train(2)
print("después de entrenamiento")

fig = sbs_rnn.plot_losses()
fig.show()

#accuracy_matrix = (StepByStep.loader_apply(test_loader, sbs_rnn.correct))
#print(accuracy_matrix)
#accuracy = [row[0]/row[1]for row in accuracy_matrix]
#print(f'Total accuracy: {np.mean(accuracy)*100} %')

noise = torch.randn(batch_size, 1, 1, device='cuda' if torch.cuda.is_available() else 'cpu').permute((0, 2, 1))
fake_data = sbs_rnn.predict(batch_size)
fake_data = fake_data.detach().cpu().numpy()
np.savetxt('data/alibaba/generated/fake_data.csv', fake_data.reshape(-1,1), delimiter=",")

# Print raw generator output
print("Raw Generator Output: ", fake_data)

fake_data_rescaled = np.reshape(scaler.inverse_transform(fake_data.reshape(-1,1)), fake_data.shape)

os.makedirs('data/alibaba/generated', exist_ok=True)
np.savetxt(f'data/alibaba/generated/batch_hidden_dim_{hidden_dim_discriminator}_num_layers_{num_layers}_glr_{generator_lr}_dlr_{discriminator_lr}.csv', fake_data_rescaled[0], delimiter=",")
