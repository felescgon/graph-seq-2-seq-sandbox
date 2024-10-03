import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Reshape
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

X_rama_2 = original_df[['start_time', 'end_time']]
X_rama_2['duration'] = X_rama_2['end_time'] - X_rama_2['start_time']
scaler_rama_2 = StandardScaler()
X_rama_2_scaled = scaler_rama_2.fit_transform(X_rama_2[['start_time', 'end_time', 'duration']])

X_rama_3 = original_df[task_type_columns + task_category_columns + job_columns]
encoder_rama_3 = OneHotEncoder(sparse_output=False)
X_rama_3_encoded = encoder_rama_3.fit_transform(X_rama_3)

def create_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

autoencoder_rama_3, encoder_rama_3 = create_autoencoder(X_rama_3_encoded.shape[1], 32)

history_rama_3 = autoencoder_rama_3.fit(X_rama_3_encoded, X_rama_3_encoded, epochs=15, batch_size=32, validation_split=0.2)

def plot_autoencoder_loss(history, title):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_autoencoder_loss(history_rama_3, 'Autoencoder 3 - Loss')

X_rama_3_encoded = encoder_rama_3.predict(X_rama_3_encoded)

input_rama_1 = Input(shape=(X_rama_1_scaled.shape[1],))
dense_rama_1 = Dense(2048, activation='relu')(input_rama_1)
dense_rama_1 = Dense(1024, activation='relu')(dense_rama_1)
dense_rama_1 = Reshape((1, 1024))(dense_rama_1)

input_rama_2 = Input(shape=(X_rama_2_scaled.shape[1],))
dense_rama_2 = Dense(2048, activation='relu')(input_rama_2)
dense_rama_2 = Dense(1024, activation='relu')(dense_rama_2)
dense_rama_2 = Reshape((1, 1024))(dense_rama_2)

input_rama_3 = Input(shape=(X_rama_3_encoded.shape[1],))
dense_rama_3 = Dense(2048, activation='relu')(input_rama_3)
dense_rama_3 = Dense(1024, activation='relu')(dense_rama_3)
dense_rama_3 = Reshape((1, 1024))(dense_rama_3)

combined = concatenate([dense_rama_1, dense_rama_2, dense_rama_3])

lstm = LSTM(512, return_sequences=False)(combined)

dense_combined = Dense(1, activation='linear')(lstm)

model = Model(inputs=[input_rama_1, input_rama_2, input_rama_3], outputs=dense_combined)
model.compile(optimizer='adam', loss='mse')

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

y = original_df['instance_num'].values

X_train_rama_1, X_test_rama_1, X_train_rama_2, X_test_rama_2, X_train_rama_3, X_test_rama_3, y_train, y_test = train_test_split(
    X_rama_1_scaled, X_rama_2_scaled, X_rama_3_encoded, y, test_size=0.2, random_state=42)

history = model.fit([X_train_rama_1, X_train_rama_2, X_train_rama_3], y_train,
                    epochs=20, batch_size=32, validation_split=0.2)

loss = model.evaluate([X_test_rama_1, X_test_rama_2, X_test_rama_3], y_test)
print(f"Loss on the testing set: {loss}")

predictions = model.predict([X_test_rama_1, X_test_rama_2, X_test_rama_3])

mse = mean_squared_error(y_test, predictions)
print(f"Mean squared error on the testing set: {mse}")

predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")
