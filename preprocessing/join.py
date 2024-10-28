
import torch
import pandas as pd
import numpy as np
import yaml
import json

from itertools import product
from pathlib import Path


def join_extracts():
    """
    Reads:
        apc_dataset.csv (corresponding to X data and metadata)
        mean_MR.csv (corresponding to y data)

    Writes:
        data_joint.csv (aligned: metadata + X data + y data)
        metadata.csv (aligned metadata)
        X_data.csv (aligned X data)
        y_data.csv (aligned y data)

        index_id_map.json (alignment map from tensor index in dim 0 to ID)
        X_col_map.json (alignment map of columns from tensor dim 1 to columns of X)
        y_col_map.json (alignment map of columns from tensor dim 1 to columns of y)

        X_data_tensor.pt (tensor of X data)
        X_data_tensor.pt (tensor of y data)
    """

    ###--- Paths and Settings ---###
    data_dir = Path("./data")
    
    time_frames = ['150823_050124', '050124_050624']

    ###--- 0. Load Raw Data ---###
    file_paths_X = [data_dir / 'raw' / f'apc_dataset_{timeframe}.csv' for timeframe in time_frames]
    file_paths_y = [data_dir / 'raw' / f'metr_MR_means_{timeframe}.csv' for timeframe in time_frames]

    raw_X_data_dfs = [pd.read_csv(X_data_path) for X_data_path in file_paths_X]
    raw_y_data_dfs = [pd.read_csv(y_data_path) for y_data_path in file_paths_y]

    raw_X_data_df = pd.concat(raw_X_data_dfs, axis = 0, ignore_index = True)
    raw_y_data_df = pd.concat(raw_y_data_dfs, axis = 0, ignore_index = True)

    print(
        f'Size result: \n'
        f'--------------------------\n'
        f'X: \n{len(raw_X_data_df)}\n' 
        f'y: \n{len(raw_y_data_df)}\n'
        f'--------------------------\n'
        f'X wo duplicates: \n{len(raw_X_data_df.drop_duplicates())}\n' 
        f'y wo duplicates: \n{len(raw_y_data_df.drop_duplicates())}\n'
    )

    raw_X_data_df = raw_X_data_df.drop_duplicates(ignore_index = True)

    org_X_data_df = pd.read_csv(data_dir / 'apc_dataset.csv')

    print(len(raw_X_data_df.columns))
    print(len(org_X_data_df.columns))
    # const_zero = lambda df, col: (df[col] == 0).all(axis = 0)

    # const_zero_cols = [col for col in raw_X_data_df if const_zero(raw_X_data_df, col)]
    
    # raw_X_data_df.drop(columns = const_zero_cols, inplace = True)

    # print(len(raw_X_data_df.columns), raw_X_data_df.columns)

