import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Reshape
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv')
original_df = pd.read_csv(file_path, header=0)

# Handle missing values
original_df = original_df.fillna(original_df.mean())

# Define the columns for each branch
plan_cpu_columns = [col for col in original_df.columns if 'plan_cpu' in col]
plan_mem_columns = [col for col in original_df.columns if 'plan_mem' in col]
task_type_columns = [col for col in original_df.columns if 'task_type' in col]
task_category_columns = [col for col in original_df.columns if 'task_category' in col]
job_columns = [col for col in original_df.columns if 'job_' in col]

# Preprocess the data for each branch
X_rama_1 = original_df[['instance_num'] + plan_cpu_columns + plan_mem_columns]
scaler_rama_1 = StandardScaler()
X_rama_1_scaled = scaler_rama_1.fit_transform(X_rama_1)

X_rama_2 = original_df[['start_time', 'end_time']]
X_rama_2['duration'] = X_rama_2['end_time'] - X_rama_2['start_time']
scaler_rama_2 = StandardScaler()
X_rama_2_scaled = scaler_rama_2.fit_transform(X_rama_2[['start_time', 'end_time', 'duration']])

X_rama_3 = original_df[task_type_columns + task_category_columns + job_columns]
encoder_rama_3 = OneHotEncoder(sparse_output=False)
X_rama_3_encoded = encoder_rama_3.fit_transform(X_rama_3)

# Function to create and return the autoencoder and encoder
def create_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model (to extract the encoded features)
    encoder = Model(input_layer, encoded)

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

# Create and train autoencoders for each branch
autoencoder_rama_1, encoder_rama_1 = create_autoencoder(X_rama_1_scaled.shape[1], 512)
autoencoder_rama_2, encoder_rama_2 = create_autoencoder(X_rama_2_scaled.shape[1], 512)
autoencoder_rama_3, encoder_rama_3 = create_autoencoder(X_rama_3_encoded.shape[1], 512)

# Train the autoencoders and store the training history for each one
history_rama_1 = autoencoder_rama_1.fit(X_rama_1_scaled, X_rama_1_scaled, epochs=50, batch_size=32, validation_split=0.2)
history_rama_2 = autoencoder_rama_2.fit(X_rama_2_scaled, X_rama_2_scaled, epochs=50, batch_size=32, validation_split=0.2)
history_rama_3 = autoencoder_rama_3.fit(X_rama_3_encoded, X_rama_3_encoded, epochs=50, batch_size=32, validation_split=0.2)

# Plot loss for each autoencoder
def plot_autoencoder_loss(history, title):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the loss for each autoencoder
plot_autoencoder_loss(history_rama_1, 'Autoencoder 1 - Loss')
plot_autoencoder_loss(history_rama_2, 'Autoencoder 2 - Loss')
plot_autoencoder_loss(history_rama_3, 'Autoencoder 3 - Loss')

# Encode the inputs using the trained encoders
X_rama_1_encoded = encoder_rama_1.predict(X_rama_1_scaled)
X_rama_2_encoded = encoder_rama_2.predict(X_rama_2_scaled)
X_rama_3_encoded = encoder_rama_3.predict(X_rama_3_encoded)

# Define the model architecture using encoded inputs
input_rama_1 = Input(shape=(X_rama_1_encoded.shape[1],))  # 2D input
dense_rama_1 = Dense(2048, activation='relu')(input_rama_1)
dense_rama_1 = Dense(1024, activation='relu')(dense_rama_1)
dense_rama_1 = Reshape((1, 1024))(dense_rama_1)  # Reshape to 3D for concatenation

input_rama_2 = Input(shape=(X_rama_2_encoded.shape[1],))  # 2D input
dense_rama_2 = Dense(2048, activation='relu')(input_rama_2)
dense_rama_2 = Dense(1024, activation='relu')(dense_rama_2)
dense_rama_2 = Reshape((1, 1024))(dense_rama_2)  # Reshape to 3D

input_rama_3 = Input(shape=(X_rama_3_encoded.shape[1],))  # 2D input
dense_rama_3 = Dense(2048, activation='relu')(input_rama_3)
dense_rama_3 = Dense(1024, activation='relu')(dense_rama_3)
dense_rama_3 = Reshape((1, 1024))(dense_rama_3)  # Reshape to 3D

# Combine the outputs of each branch
combined = concatenate([dense_rama_1, dense_rama_2, dense_rama_3])

# Add LSTM layer to capture the temporal dependencies
lstm = LSTM(512, return_sequences=False)(combined)

# Output layer
dense_combined = Dense(1, activation='linear')(lstm)

# Define the model
model = Model(inputs=[input_rama_1, input_rama_2, input_rama_3], outputs=dense_combined)
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()

# Define the target variable
y = original_df['instance_num'].values

# Split the data into training and testing sets
X_train_rama_1, X_test_rama_1, X_train_rama_2, X_test_rama_2, X_train_rama_3, X_test_rama_3, y_train, y_test = train_test_split(
    X_rama_1_encoded, X_rama_2_encoded, X_rama_3_encoded, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit([X_train_rama_1, X_train_rama_2, X_train_rama_3], y_train,
                    epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing set
loss = model.evaluate([X_test_rama_1, X_test_rama_2, X_test_rama_3], y_test)
print(f"Loss on the testing set: {loss}")

# Make predictions on the testing set
predictions = model.predict([X_test_rama_1, X_test_rama_2, X_test_rama_3])

# Evaluate the performance of the model using mean squared error
mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error on the testing set: {mse}")

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
