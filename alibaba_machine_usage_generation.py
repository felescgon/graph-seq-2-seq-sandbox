import torch
import torch.nn as nn
import torch.optim as optim
from helpers import index_splitter
from helpers import make_balanced_sampler
from data_load import get_alibaba_machine_usage
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from Discriminator import Discriminator
from Generator import Generator
from trainer import StepByStep
from sklearn.preprocessing import StandardScaler
import numpy as np

seq_len = 360
x, y = get_alibaba_machine_usage(5000, seq_len)
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
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)

num_layers = 2
rnn_layer = nn.LSTM
bidirectional = False
n_discriminator_outputs = 1
n_features = x_train_tensor.shape[2]
hidden_dim_discriminator = 24
hidden_dim_generator = 48

batch_norm = True
dropout = 0.3

torch.manual_seed(21)
generator = Generator(n_features=n_features, hidden_dim=hidden_dim_generator, sequence_length=seq_len, num_layers=num_layers, bidirectional=bidirectional, rnn_layer=rnn_layer, batch_norm=batch_norm, dropout=dropout)
discriminator = Discriminator(n_features=n_features, hidden_dim=hidden_dim_discriminator, n_outputs=n_discriminator_outputs, num_layers=num_layers, bidirectional=bidirectional, rnn_layer=rnn_layer, batch_norm=batch_norm)
loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters())
discriminator_optimizer = optim.Adam(discriminator.parameters())

sbs_rnn = StepByStep(generator, discriminator, loss, generator_optimizer, discriminator_optimizer)
sbs_rnn.set_loaders(train_loader, test_loader)
print("antes de entrenamiento")
sbs_rnn.train(2)
print("después de entrenamiento")

fig = sbs_rnn.plot_losses()
fig.show()
# accuracy_matrix = (StepByStep.loader_apply(test_loader, sbs_rnn.correct))
# print(accuracy_matrix)
# accuracy = [row[0]/row[1]for row in accuracy_matrix]
# print(f'Total accuracy: {np.mean(accuracy)*100} %')
batch_size = 24*4
noise = torch.randn((batch_size, seq_len, n_features), device='cuda' if torch.cuda.is_available() else 'cpu')
fake_data = sbs_rnn.predict()
fake_data_rescaled = np.reshape(scaler.inverse_transform(fake_data.reshape(-1,1)), fake_data.shape)
for index, batch in enumerate(fake_data_rescaled,1):
    np.savetxt(f'data/alibaba/generated/batch_{index}.csv', fake_data_rescaled[0], delimiter=",")
