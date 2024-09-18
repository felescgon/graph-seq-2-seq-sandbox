import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Sample data
data = pd.DataFrame({
    'instance_num': [40, 1, 1],
    'start_time': [86207, 86207, 86207],
    'end_time': [86485, 86622, 86210],
    'plan_cpu': [50.0, 50.0, 50.0],
    'plan_mem': [0.39, 0.2, 0.2],
    'main_task': [2, 7, 3],
    'task_type_1': [1, 1, 1],
    # Add other columns as per your dataset
})

# Normalize the float and integer columns
float_cols = ['start_time', 'end_time', 'plan_cpu', 'plan_mem']
scaler = MinMaxScaler()
data[float_cols] = scaler.fit_transform(data[float_cols])

# Create autoencoder for one-hot encoded features
one_hot_features = data.iloc[:, 5:].values

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

input_dim = one_hot_features.shape[1]
encoding_dim = 10  # Adjust this based on your needs

autoencoder = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert to PyTorch tensors
one_hot_tensor = torch.FloatTensor(one_hot_features)
dataloader = DataLoader(TensorDataset(one_hot_tensor, one_hot_tensor), batch_size=256, shuffle=True)

# Train the autoencoder
num_epochs = 50
for epoch in range(num_epochs):
    for batch_data in dataloader:
        inputs, _ = batch_data
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encode the one-hot features
encoded_features = autoencoder.encoder(one_hot_tensor).detach().numpy()

# Combine encoded one-hot features with normalized float values
combined_features = np.hstack((data[float_cols].values, encoded_features))

# Create a new autoencoder for the combined features
combined_input_dim = combined_features.shape[1]
combined_encoding_dim = 10  # Adjust this based on your needs

combined_autoencoder = Autoencoder(combined_input_dim, combined_encoding_dim)
optimizer = optim.Adam(combined_autoencoder.parameters(), lr=0.001)

# Convert to PyTorch tensors
combined_tensor = torch.FloatTensor(combined_features)
combined_dataloader = DataLoader(TensorDataset(combined_tensor, combined_tensor), batch_size=256, shuffle=True)

# Train the combined autoencoder
for epoch in range(num_epochs):
    for batch_data in combined_dataloader:
        inputs, _ = batch_data
        outputs = combined_autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encode and decode the combined features
encoded_combined_features = combined_autoencoder.encoder(combined_tensor).detach().numpy()
decoded_combined_features = combined_autoencoder.decoder(combined_autoencoder.encoder(combined_tensor)).detach().numpy()

# Denormalize the float columns
denormalized_floats = scaler.inverse_transform(decoded_combined_features[:, :len(float_cols)])

# Decode the one-hot encoded features
decoded_one_hot = autoencoder.decoder(torch.FloatTensor(decoded_combined_features[:, len(float_cols):])).detach().numpy()

# Combine the denormalized and decoded features
final_decoded_features = np.hstack((denormalized_floats, decoded_one_hot))

# Create DataFrames for encoded and decoded data
encoded_df = pd.DataFrame(encoded_combined_features, columns=[f'encoded_{i}' for i in range(combined_encoding_dim)])
decoded_columns = float_cols + data.columns[5:].tolist()
decoded_df = pd.DataFrame(final_decoded_features, columns=decoded_columns)

# Save the dataframes to CSV files
encoded_df.to_csv('encoded_features.csv', index=False, header=True)
decoded_df.to_csv('decoded_features.csv', index=False, header=True)
