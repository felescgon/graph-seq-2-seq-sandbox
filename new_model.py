import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

file_path = os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv')
original_df = pd.read_csv(file_path, header=0)
original_df = original_df.fillna(original_df.mean())

plan_cpu_columns = [col for col in original_df.columns if 'plan_cpu' in col]
plan_mem_columns = [col for col in original_df.columns if 'plan_mem' in col]
task_type_columns = [col for col in original_df.columns if 'task_type' in col]
task_category_columns = [col for col in original_df.columns if 'task_category' in col]
job_columns = [col for col in original_df.columns if 'job_' in col]

X_rama_1 = original_df[['instance_num'] + plan_cpu_columns + plan_mem_columns]
scaler_rama_1 = StandardScaler()
X_rama_1_scaled = scaler_rama_1.fit_transform(X_rama_1)

X_rama_2 = original_df[['start_time', 'end_time', 'duration']]
scaler_rama_2 = StandardScaler()
X_rama_2_scaled = scaler_rama_2.fit_transform(X_rama_2[['start_time', 'end_time', 'duration']])

X_rama_3 = original_df[task_type_columns + task_category_columns + job_columns]

# TODO: Añadir encoder decoder para el resto de features
class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

encoder_decoder_rama_3 = EncoderDecoder(X_rama_3.shape[1], 32)
criterion = nn.MSELoss()
optimizer = optim.Adam(encoder_decoder_rama_3.parameters(), lr=0.001)

X_rama_3_tensor = torch.FloatTensor(X_rama_3.values)

num_epochs = 50
batch_size = 32
num_batches = len(X_rama_3_tensor) // batch_size

train_losses = []
val_losses = []


for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
    encoder_decoder_rama_3.train()
    epoch_train_loss = 0
    for i in range(num_batches):
        batch = X_rama_3_tensor[i * batch_size:(i + 1) * batch_size]
        optimizer.zero_grad()
        outputs, _ = encoder_decoder_rama_3(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= num_batches
    train_losses.append(epoch_train_loss)

    encoder_decoder_rama_3.eval()
    with torch.no_grad():
        val_outputs, _ = encoder_decoder_rama_3(X_rama_3_tensor)
        val_loss = criterion(val_outputs, X_rama_3_tensor)
        val_losses.append(val_loss.item())

    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}, Validation Loss: {val_loss.item():.4f}")

plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


with torch.no_grad():
    X_rama_3_encoded = encoder_decoder_rama_3.encoder(X_rama_3_tensor)

X = np.concatenate([X_rama_1_scaled, X_rama_2_scaled, X_rama_3_encoded.numpy()], axis=1)
y = np.roll(X, -1, axis=0)

X = X[:-1]
y = y[:-1]

X_train_rama_1, X_test_rama_1, X_train_rama_2, X_test_rama_2, X_train_rama_3, X_test_rama_3, y_train, y_test = train_test_split(
    X_rama_1_scaled[:-1], X_rama_2_scaled[:-1], X_rama_3_encoded[:-1].numpy(), y, test_size=0.2, random_state=42)

class LSTMModel(nn.Module):
    def __init__(self, input_dim_rama_1, input_dim_rama_2, input_dim_rama_3, output_size):
        super(LSTMModel, self).__init__()
        self.dense_rama_1 = nn.Sequential(
            nn.Linear(input_dim_rama_1, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        self.dense_rama_2 = nn.Sequential(
            nn.Linear(input_dim_rama_2, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        self.dense_rama_3 = nn.Sequential(
            nn.Linear(input_dim_rama_3, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(3072, 512)
        self.output_layer = nn.Linear(512, output_size)

    def forward(self, rama_1, rama_2, rama_3):
        dense_out_1 = self.dense_rama_1(rama_1).unsqueeze(1)
        dense_out_2 = self.dense_rama_2(rama_2).unsqueeze(1)
        dense_out_3 = self.dense_rama_3(rama_3).unsqueeze(1)
        
        combined = torch.cat([dense_out_1, dense_out_2, dense_out_3], dim=1)

        lstm_out, _ = self.lstm(combined)
        
        return self.output_layer(lstm_out[:, -1, :])

input_size_rama_1 = X_rama_1_scaled.shape[1]
input_size_rama_2 = X_rama_2_scaled.shape[1]
input_size_rama_3 = X_rama_3_encoded.shape[1]
output_size = X_rama_1_scaled.shape[1] + X_rama_2_scaled.shape[1] + X_rama_3_encoded.shape[1]

model = LSTMModel(input_size_rama_1, input_size_rama_2, input_size_rama_3, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_train_rama_1_tensor = torch.FloatTensor(X_train_rama_1)
X_train_rama_2_tensor = torch.FloatTensor(X_train_rama_2)
X_train_rama_3_tensor = torch.FloatTensor(X_train_rama_3)
y_train_tensor = torch.FloatTensor(y_train)

X_test_rama_1_tensor = torch.FloatTensor(X_test_rama_1)
X_test_rama_2_tensor = torch.FloatTensor(X_test_rama_2)
X_test_rama_3_tensor = torch.FloatTensor(X_test_rama_3)
y_test_tensor = torch.FloatTensor(y_test)

train_losses = []
val_losses = []

num_epochs = 100
for epoch in tqdm(range(num_epochs), desc="Training Epochs", unit="epoch"):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_rama_1_tensor, X_train_rama_2_tensor, X_train_rama_3_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    train_loss = loss.item()
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_rama_1_tensor, X_test_rama_2_tensor, X_test_rama_3_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)

    val_losses.append(val_loss.item())
    tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss.item():.4f}")


model.eval()
X_test_rama_1_tensor = torch.FloatTensor(X_test_rama_1)
X_test_rama_2_tensor = torch.FloatTensor(X_test_rama_2)
X_test_rama_3_tensor = torch.FloatTensor(X_test_rama_3)
y_test_tensor = torch.FloatTensor(y_test)



with torch.no_grad():
    predictions = model(X_test_rama_1_tensor, X_test_rama_2_tensor, X_test_rama_3_tensor)

loss = criterion(predictions, y_test_tensor)
print(f"Loss on the test set: {loss.item()}")

mse = mean_squared_error(y_test, predictions.numpy())
print(f"Mean squared error on the testing set: {mse}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(1, num_epochs + 1))
plt.legend()
plt.grid()
plt.show()

predictions_numerical_1 = predictions[:, :X_rama_1_scaled.shape[1]]
predictions_numerical_2 = predictions[:, X_rama_1_scaled.shape[1]:X_rama_1_scaled.shape[1] + X_rama_2_scaled.shape[1]]
predictions_categorical = predictions[:, -X_rama_3_encoded.shape[1]:]

predictions_numerical_1_original = scaler_rama_1.inverse_transform(predictions_numerical_1.numpy())
predictions_numerical_2_original = scaler_rama_2.inverse_transform(predictions_numerical_2.numpy())
predictions_categorical_original = encoder_decoder_rama_3.decoder(torch.FloatTensor(predictions_categorical.detach().numpy())).detach().numpy()

predictions_original = np.concatenate([predictions_numerical_1_original, predictions_numerical_2_original, predictions_categorical_original], axis=1)

predictions_df = pd.DataFrame(predictions_original, columns=original_df.columns)
predictions_df.to_csv('predictions_original.csv', index=False)

print("Destandardized and decoded predictions saved to 'predictions_original.csv'")
