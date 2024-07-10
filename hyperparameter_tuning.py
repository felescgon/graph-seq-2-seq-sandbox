import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
from tqdm import tqdm
import pandas as pd

from data_load import get_alibaba_encoded_machine_usage
from Discriminator import Discriminator
from Generator import Generator
from helpers import index_splitter
from trainer import StepByStep

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Hyperparameter ranges
ngf_values = [32, 64, 128]
hidden_dim_values = [16, 32, 64, 128]
lr_values = [0.0001, 0.0005, 0.001]
num_layers_values = [2, 3, 4, 5]
dropout_values = [0.0, 0.2, 0.5]
num_epochs = 10
batch_size = 128

# Prepare data
seq_len = 64
x, y, autoencoder = get_alibaba_encoded_machine_usage(5120, seq_len)
train_idx, val_idx = index_splitter(len(x), [80, 20])
x_tensor = torch.as_tensor(x).float()
y_tensor = torch.as_tensor(y).float()

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

train_dataset = TensorDataset(scaled_x_train_tensor.float(), y_train_tensor.view(-1, 1).float())
val_dataset = TensorDataset(scaled_x_val_tensor.float(), y_val_tensor.view(-1, 1).float())

results = []

# Initialize tqdm for the hyperparameter tuning loop
for ngf, hidden_dim, lr, num_layers, dropout in tqdm(list(product(ngf_values, hidden_dim_values, lr_values, num_layers_values, dropout_values)), desc="Hyperparameter Tuning"):
    #print(f'Training with ngf={ngf}, hidden_dim={hidden_dim}, lr={lr}, num_layers={num_layers}, dropout={dropout}')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    generator = Generator(sequence_length=seq_len, n_features=x_tensor.shape[2], ngf=ngf).to('cuda' if torch.cuda.is_available() else 'cpu')
    generator.apply(weights_init)
    
    discriminator = Discriminator(n_features=x_tensor.shape[2], hidden_dim=hidden_dim, n_outputs=1, num_layers=num_layers, dropout=dropout).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    
    loss_fn = nn.BCELoss()
    
    sbs = StepByStep(generator, discriminator, loss_fn, generator_optimizer, discriminator_optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(num_epochs)
    
    # Evaluate and save results
    val_loss = sbs._mini_batch(validation=True)
    results.append((ngf, hidden_dim, lr, num_layers, dropout, val_loss))
    #print(f'Validation loss: {val_loss}')

    fake_data = sbs.predict(10)
    fake_data_rescaled = np.reshape(scaler.inverse_transform(fake_data.reshape(-1,1)), fake_data.shape)
    
    os.makedirs('data/alibaba/generated', exist_ok=True)

    decoded_samples = []
    for i in range(fake_data_rescaled.shape[0]):
        sample = fake_data_rescaled[i]
        
        # Extract the last 8 features of the sample
        reshaped_input = torch.tensor(sample[:, -8:], dtype=torch.float32)
        
        # Pass the reshaped input through the decoder
        decoded_sample = autoencoder.decoder(reshaped_input).detach().numpy()
        
        # Combine the decoded data with the rest of the rescaled data for the sample
        combined_sample = np.concatenate((sample[:, :-8], decoded_sample), axis=1)
        
        # Append the processed sample to the list
        decoded_samples.append(combined_sample)
        
        # Save the processed sample to a CSV file
        os.makedirs(f'data/alibaba/generated/hidden_dim_{hidden_dim}_num_layers_{num_layers}_lr_{lr}_dropout_{dropout}_ngf_{ngf}', exist_ok=True)

        sample_file_path = os.path.join(f'data/alibaba/generated/hidden_dim_{hidden_dim}_num_layers_{num_layers}_lr_{lr}_dropout_{dropout}_ngf_{ngf}/sample_{i}.csv')
        np.savetxt(sample_file_path, combined_sample, delimiter=",")

# Find the best hyperparameters
best_hyperparams = min(results, key=lambda x: x[-1])
#print(f'Best hyperparameters: ngf={best_hyperparams[0]}, hidden_dim={best_hyperparams[1]}, lr={best_hyperparams[2]}, num_layers={best_hyperparams[3]}, dropout={best_hyperparams[4]}, validation loss={best_hyperparams[5]}')

# Save results to a csv file
results_file = 'hyperparameter_tuning_results.csv'
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ngf', 'hidden_dim', 'lr', 'num_layers', 'dropout', 'val_loss'])
    for res in results:
        writer.writerow([res[0], res[1], res[2], res[3], res[4], res[5]])
