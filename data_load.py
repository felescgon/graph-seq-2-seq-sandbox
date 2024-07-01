import os
import pandas as pd
import numpy as np

np.random.seed(42)

def get_alibaba_machine_usage(n_samples, sequence_length):
    np.random.seed(42)

    original_df = pd.read_csv(os.path.abspath('../graph-seq2seq/data/alibaba/batch_task_chunk_preprocessed_10000.csv'), header=0)
    #original_df = pd.read_csv(os.path.abspath('../graph-seq2seq/data/alibaba/original_4_days.csv'))
    original_idx = np.arange(original_df.shape[0]-sequence_length)

    np.random.shuffle(original_idx)

    splitted_original_x = np.array([original_df[index:index+sequence_length] for index in original_idx[:n_samples]])
    splitted_original_y = np.ones(len(splitted_original_x))

    return splitted_original_x, splitted_original_y