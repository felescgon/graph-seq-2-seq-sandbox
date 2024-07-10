import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_alibaba_encoded_machine_usage(n_samples, sequence_length):
    np.random.seed(42)

    original_df = pd.read_csv(os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv'), header=0)
    original_df, autoencoder = autoencoder_encode(original_df)

    original_idx = np.arange(original_df.shape[0]-sequence_length)

    np.random.shuffle(original_idx)

    splitted_original_x = np.array([original_df[index:index+sequence_length] for index in original_idx[:n_samples]])
    splitted_original_y = np.ones(len(splitted_original_x))

    return splitted_original_x, splitted_original_y, autoencoder

def autoencoder_encode(original_df):
    np.random.seed(42)

    features = original_df.drop(columns=['instance_num', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'main_task'])

    # Convert data to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, features_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = features.shape[1]
    encoding_dim = 8  # Adjust based on your needs
    autoencoder = Autoencoder(input_dim, encoding_dim)

    # Training the autoencoder
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 100

    for _ in range(num_epochs):
        for data in dataloader:
            inputs, _ = data
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Encode the data
    encoded_data = autoencoder.encoder(features_tensor).detach().numpy()

    # Reconstruct the original DataFrame with decoded features
    encoded_columns = [f'encoded_dim_{i+1}' for i in range(encoding_dim)]
    encoded_data = pd.DataFrame(encoded_data, columns=encoded_columns)
    final_df = pd.concat([original_df[['instance_num', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'main_task']], encoded_data], axis=1)

    return final_df, autoencoder
