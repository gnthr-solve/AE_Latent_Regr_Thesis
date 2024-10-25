

import torch
import pandas as pd
import numpy as np
import yaml
import json

from itertools import product
from pathlib import Path

from .info import file_metadata_cols, process_metadata_cols, identifier_col, y_col_rename_map

"""
Preprocessing - I. Preprocessing Raw DataFrames to Organised pytorch Tensors
-------------------------------------------------------------------------------------------------------------------------------------------
    0. Load Raw Data
    1. Eliminate File Metadata and Rename
    2. Extract Column/Parameter Names for X, y
        2.1. Create Column Maps
        2.2. Export Column Maps
    3. Merge X and y DataFrames on Identifier
        3.1. Eliminate Rows Containing NaN in X Columns
        3.2. Divide y values by time
    4. Create Index Map and Export
        4.1. Export Index Map
        4.2. Export Joint DataFrame
    5. Separate and Export DataFrames
        5.1. Split Joint DataFrame
        5.2. Export DataFrames
    6. Convert to Tensors and Export
"""

def preprocess_raw():
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
    tensor_one_mapping = True
    data_dir = Path("./data")
    alignment_info_dir = data_dir / "alignment_info"
    tensor_dir = data_dir / "tensors"

    joint_data_df_name = "data_joint.csv"
    metadata_df_name = "metadata.csv"
    X_data_df_name = "X_data.csv"
    y_data_df_name = "y_data.csv"

    X_data_tensor_name = 'X_data_tensor.pt'
    y_data_tensor_name = 'y_data_tensor.pt'


    ###--- 0. Load Raw Data ---###
    raw_X_data_path = data_dir / "apc_dataset.csv"
    raw_y_data_path = data_dir / "mean_MR.csv"

    raw_X_data_df = pd.read_csv(raw_X_data_path)
    raw_y_data_df = pd.read_csv(raw_y_data_path)


    ###--- 1. Eliminate File Metadata and Rename ---###
    X_data_df = raw_X_data_df.drop(columns = file_metadata_cols)

    y_data_df = raw_y_data_df.rename(columns = y_col_rename_map)
    

    ###--- 2. Extract Column/Parameter Names for X, y ---###
    X_data_cols = X_data_df.drop(columns = [identifier_col] + process_metadata_cols).columns.tolist()
    y_data_cols = y_data_df.drop(columns = [identifier_col]).columns.tolist()

    #--- 2.1. Create Column Maps ---#
    map_incr = 1 if tensor_one_mapping else 0

    X_col_map = {0: 'mapping_idx'}
    X_col_map.update({
        i + map_incr: col
        for i, col in enumerate(X_data_cols)
    })

    y_col_map = {0: 'mapping_idx'}
    y_col_map.update({
        i + map_incr: col
        for i, col in enumerate(y_data_cols)
    })

    #--- 2.2. Export Column Maps ---#
    tensor_infix = '_tensor' if tensor_one_mapping else ''

    with open(alignment_info_dir / f'X{tensor_infix}_col_map.json', 'w') as f:
        json.dump(X_col_map, f)

    with open(alignment_info_dir / f'y{tensor_infix}_col_map.json', 'w') as f:
        json.dump(y_col_map, f)


    ###--- 3. Merge X and y DataFrames on Identifier ---###
    data_df = X_data_df.merge(y_data_df, on = [identifier_col], how = 'left')

    #--- 3.1. Eliminate Rows Containing NaN in X Columns ---#
    isna_any_mask = data_df[X_data_cols].isna().any(axis = 1)
    data_df = data_df[~isna_any_mask]
    data_df.reset_index(inplace = True)

    #--- 3.2. Divide y-values by Time to get MRR ---#
    data_df[y_data_cols] = data_df[y_data_cols].div(data_df['Time_ptp'], axis = 0)


    ###--- 4. Create Index Map and Export ---###
    index_id_map = data_df[identifier_col].to_dict()
    data_df.insert(0, 'mapping_idx', data_df.index) 

    #--- 4.1. Export Index Map ---#
    with open(alignment_info_dir / 'index_id_map.json', 'w') as f:
        json.dump(index_id_map, f)
    
    #--- 4.2. Export Joint DataFrame ---#
    data_df.to_csv(data_dir / joint_data_df_name, index=False)


    ###--- 5. Separate and Export DataFrames ---###
    X_data_cols = list(X_col_map.values())
    y_data_cols = list(y_col_map.values())
    
    metadata_cols = ['mapping_idx', identifier_col] + process_metadata_cols \
                    if tensor_one_mapping else [identifier_col] + process_metadata_cols
    
    #--- 5.1. Split Joint DataFrame ---#
    metadata_df = data_df[metadata_cols]
    X_data_df = data_df[X_data_cols]
    y_data_df = data_df[y_data_cols]

    #--- 5.2. Export DataFrames ---#
    metadata_df.to_csv(data_dir / metadata_df_name, index=False)
    X_data_df.to_csv(data_dir / X_data_df_name, index=False)
    y_data_df.to_csv(data_dir / y_data_df_name, index=False)


    ###--- 6. Convert to Tensors and Export ---###
    X_data_tensor = torch.tensor(X_data_df.to_numpy(), dtype=torch.float32)
    y_data_tensor = torch.tensor(y_data_df.to_numpy(), dtype=torch.float32)

    torch.save(X_data_tensor, tensor_dir / X_data_tensor_name)
    torch.save(y_data_tensor, tensor_dir / y_data_tensor_name)