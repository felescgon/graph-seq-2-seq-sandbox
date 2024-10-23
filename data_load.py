import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datacentertracesdatasets import loadtraces
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_ori_data(sequence_length=60, stride=1, shuffle=False, seed=13, ori_data_filename=None, input_output_ratio=0.5, trace='azure_v2'):
    np.random.seed(seed)
    if ori_data_filename is None:
        original_df = loadtraces.get_trace(trace_name=trace, trace_type='machine_usage', stride_seconds=300, format='dataframe')
    else:
        original_df = pd.read_csv(ori_data_filename, header=None)
    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))
    if shuffle:
        np.random.shuffle(start_sequence_range)
    split_size = int(sequence_length * input_output_ratio)
    splitted_original_data = np.array([original_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    splitted_original_x = splitted_original_data[:, :split_size]
    splitted_original_y = splitted_original_data[:, split_size:]
    return splitted_original_x, splitted_original_y


def scale_data(ori_data, scaling_method='standard', scaler=None):
    reshaped_ori_data = ori_data.reshape(-1, 1)
    if scaler is None:
        assert scaling_method in ['standard', 'minmax'], 'Only standard and minmax scalers are currently supported'
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.data_min_, scaler.data_max_]

        elif scaling_method == 'standard':
            scaler = StandardScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.mean_, scaler.var_]
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, scaler_params
    else:
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, None
    
def get_data_loaders(scaled_x_train_tensor, scaled_x_val_tensor, seq_len, batch_size, input_output_ratio):
    train_data = TensorDataset(scaled_x_train_tensor.float(), scaled_x_train_tensor[:, int(seq_len / 2):].float())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    source_test = scaled_x_val_tensor[:, :int(seq_len * input_output_ratio)]
    target_test = scaled_x_val_tensor[:, int(seq_len * input_output_ratio):]
    test_data = TensorDataset(source_test.float(), target_test.float())
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_data, test_data
