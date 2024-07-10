import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Define the generator model
class Generator(nn.Module):
    def __init__(self, sequence_length, n_features, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf, n_features + 1, 4, 2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()  # Activation for one-hot encoded part
        self.tanh = nn.Tanh()  # Activation for the float part

    def forward(self, X):
        out = self.main(X).permute(0, 2, 1)
        one_hot_encoded = torch.round(self.sigmoid(out[:, :, :-1]))  # Apply sigmoid and round for one-hot encoding
        float_values = self.tanh(out[:, :, -1])  # Apply tanh for float values
        combined = torch.cat((one_hot_encoded, float_values.unsqueeze(-1)), dim=-1)  # Combine the outputs
        return combined

# Define the discriminator model
class Discriminator(nn.Module):
    def __init__(self, n_features, hidden_dim, n_outputs=1, rnn_layer=nn.LSTM, batch_norm=True, **kwargs):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features + 1  # Account for the additional float value column
        self.n_outputs = n_outputs
        self.hidden = None
        self.cell = None
        self.basic_rnn = rnn_layer(self.n_features, self.hidden_dim, batch_first=True, **kwargs)
        self.bn_hidden = nn.BatchNorm1d(hidden_dim)
        if batch_norm:
            self.bn_x = nn.BatchNorm1d(self.n_features)  # Adjust for the additional column
        else:
            self.bn_x = None

        output_dim = self.hidden_dim
        self.classifier = nn.Linear(output_dim, self.n_outputs)

    def forward(self, X):
        if self.bn_x is not None:
            X = self.bn_x(X.permute(0, 2, 1)).permute(0, 2, 1)

        rnn_out, self.hidden = self.basic_rnn(X)

        if self.bn_hidden is not None:
            rnn_out = self.bn_hidden(rnn_out.permute(0, 2, 1)).permute(0, 2, 1)

        if isinstance(self.basic_rnn, nn.LSTM):
            self.hidden, self.cell = self.hidden

        last_output = rnn_out[:, -1, :]
        out = self.classifier(last_output)

        return torch.sigmoid(out.view(-1, self.n_outputs))

# Function to load and preprocess data from CSV
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path, header=0)
    num_numeric = 7
    seq_length = df.shape[1] - num_numeric
    numeric_data = df.iloc[:, :num_numeric].values
    one_hot_data = df.iloc[:, num_numeric:].values
    return torch.tensor(numeric_data, dtype=torch.float32), torch.tensor(one_hot_data, dtype=torch.float32), seq_length

# Training the GAN
def train_gan(generator, discriminator, numeric_data, one_hot_data, seq_length, latent_dim, epochs=10000, batch_size=32, sample_interval=1000):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Data parameters
    data_size = numeric_data.size(0)

    for epoch in range(epochs):
        # Train the discriminator

        # Select a random batch of real sequences
        idx = np.random.randint(0, data_size, batch_size)
        X_numeric_real = numeric_data[idx]
        X_one_hot_real = one_hot_data[idx]
        y_real = torch.ones((batch_size, 1))

        # Concatenate numeric and one-hot real data
        X_real = torch.cat((X_numeric_real, X_one_hot_real), dim=1).unsqueeze(1)

        # Generate a batch of fake sequences
        noise = torch.randn(batch_size, latent_dim, 1)
        X_fake = generator(noise)
        X_numeric_fake = X_fake[:, :, :X_numeric_real.size(1)]
        X_one_hot_fake = X_fake[:, :, X_numeric_real.size(1):]

        # Concatenate numeric and one-hot fake data
        X_fake = torch.cat((X_numeric_fake, X_one_hot_fake), dim=2)
        y_fake = torch.zeros((batch_size, 1))

        # Discriminator loss on real and fake data
        real_loss = adversarial_loss(discriminator(X_real.squeeze(1)), y_real)
        fake_loss = adversarial_loss(discriminator(X_fake), y_fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        noise = torch.randn(batch_size, latent_dim, 1)
        valid_labels = torch.ones((batch_size, 1))

        X_fake = generator(noise)
        X_numeric_fake = X_fake[:, :, :X_numeric_real.size(1)]
        X_one_hot_fake = X_fake[:, :, X_numeric_real.size(1):]

        # Concatenate numeric and one-hot fake data
        X_fake = torch.cat((X_numeric_fake, X_one_hot_fake), dim=2)

        g_loss = adversarial_loss(discriminator(X_fake), valid_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Print the progress
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# Generate sequences and write to CSV
def generate_sequences_and_save_to_csv(generator, latent_dim, n_sequences, seq_length, num_numeric, file_path):
    noise = torch.randn(n_sequences, latent_dim, 1)
    X_fake = generator(noise)
    numeric_part = X_fake[:, :, :num_numeric]
    one_hot_logits = X_fake[:, :, num_numeric:]
    
    sequences = []
    for i in range(n_sequences):
        sequence = np.zeros(seq_length)
        numeric_values = numeric_part[i].detach().numpy()
        idx_2 = np.argmax(one_hot_logits[i].detach().numpy())
        sequence[idx_2] = 2
        logits_excluding_2 = np.delete(one_hot_logits[i].detach().numpy(), idx_2)
        indices = list(range(seq_length))
        indices.remove(idx_2)
        probabilities = np.exp(logits_excluding_2) / np.sum(np.exp(logits_excluding_2))
        chosen_indices = np.random.choice(indices, size=np.random.randint(1, seq_length), replace=False, p=probabilities)
        sequence[chosen_indices] = 1
        full_sequence = np.concatenate((numeric_values, sequence))
        sequences.append(full_sequence)
    
    df = pd.DataFrame(sequences)
    df.to_csv(file_path, index=False)

# Parameters
latent_dim = 100  # Dimensionality of the latent space
n_sequences = 100  # Number of sequences to generate
csv_path = os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv')

# Load the data
numeric_data, one_hot_data, seq_length = load_data_from_csv(csv_path)
num_numeric = numeric_data.shape[1]

# Build the generator and discriminator
generator = Generator(seq_length, num_numeric, latent_dim)
discriminator = Discriminator(num_numeric, 64)

# Train the GAN
train_gan(generator, discriminator, numeric_data, one_hot_data, seq_length, latent_dim)

# Generate sequences and save to CSV
file_path = 'generated_sequences.csv'
generate_sequences_and_save_to_csv(generator, latent_dim, n_sequences, seq_length, num_numeric, file_path)

print(f"Generated sequences saved to {file_path}")
