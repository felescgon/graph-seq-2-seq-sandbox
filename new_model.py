import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate
from sklearn.model_selection import train_test_split

file_path = os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv')
original_df = pd.read_csv(file_path, header=0)

print(original_df.head())

plan_cpu_columns = [col for col in original_df.columns if 'plan_cpu' in col]
plan_mem_columns = [col for col in original_df.columns if 'plan_mem' in col]

rama_1_columns = ['instance_num'] + plan_cpu_columns + plan_mem_columns
X_rama_1 = original_df[rama_1_columns]

scaler_rama_1 = StandardScaler()
X_rama_1_scaled = scaler_rama_1.fit_transform(X_rama_1)

X_rama_2 = original_df[['start_time', 'end_time']]
X_rama_2.loc[:, 'duration'] = X_rama_2['end_time'] - X_rama_2['start_time']

scaler_rama_2 = StandardScaler()
X_rama_2_scaled = scaler_rama_2.fit_transform(X_rama_2)

task_type_columns = [col for col in original_df.columns if 'task_type' in col]
task_category_columns = [col for col in original_df.columns if 'task_category' in col]
job_columns = [col for col in original_df.columns if 'job_' in col]

rama_3_columns = task_type_columns + task_category_columns + job_columns
X_rama_3 = original_df[rama_3_columns]

encoder_rama_3 = OneHotEncoder(sparse_output=False)
X_rama_3_encoded = encoder_rama_3.fit_transform(X_rama_3)

input_rama_1 = Input(shape=(X_rama_1_scaled.shape[1],))
dense_rama_1 = Dense(128, activation='relu')(input_rama_1)
dense_rama_1 = Dense(64, activation='relu')(dense_rama_1)

input_rama_2 = Input(shape=(X_rama_2_scaled.shape[1],))
dense_rama_2 = Dense(128, activation='relu')(input_rama_2)
dense_rama_2 = Dense(64, activation='relu')(dense_rama_2)

input_rama_3 = Input(shape=(X_rama_3_encoded.shape[1],))
dense_rama_3 = Dense(128, activation='relu')(input_rama_3)
dense_rama_3 = Dense(64, activation='relu')(dense_rama_3)

combined = concatenate([dense_rama_1, dense_rama_2, dense_rama_3])

dense_combined = Dense(128, activation='relu')(combined)
dense_combined = Dense(64, activation='relu')(dense_combined)

output = Dense(1, activation='linear')(dense_combined)

model = Model(inputs=[input_rama_1, input_rama_2, input_rama_3], outputs=output)
model.compile(optimizer='adam', loss='mse')

model.summary()

y = original_df['instance_num']

rama_1_columns = plan_cpu_columns + plan_mem_columns
X_rama_1 = original_df[rama_1_columns]

X_rama_1_scaled = scaler_rama_1.fit_transform(X_rama_1)

X_train_rama_1, X_test_rama_1, X_train_rama_2, X_test_rama_2, X_train_rama_3, X_test_rama_3, y_train, y_test = train_test_split(
    X_rama_1_scaled, X_rama_2_scaled, X_rama_3_encoded, y, test_size=0.2, random_state=42)

history = model.fit([X_train_rama_1, X_train_rama_2, X_train_rama_3], y_train, 
                    epochs=100, batch_size=32, validation_split=0.2)

loss = model.evaluate([X_test_rama_1, X_test_rama_2, X_test_rama_3], y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")

predicciones = model.predict([X_test_rama_1, X_test_rama_2, X_test_rama_3])

print(predicciones)
