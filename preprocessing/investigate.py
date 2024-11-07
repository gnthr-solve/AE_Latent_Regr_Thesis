

import torch
import pandas as pd
import numpy as np
import yaml
import json

from torch import Tensor
from itertools import product
from pathlib import Path

from helper_tools import default_index_map, default_X_col_map, default_y_col_map

"""
Investigate - Normalised Tensor
-------------------------------------------------------------------------------------------------------------------------------------------
Min-Max normalisation produces NaN entries.

Suspicion: Columns are constant 0
NOTE: Suspicion is correct
"""
def identify_min_max_NaN_source(X: Tensor):

    X_isnan = X.isnan()

    #verify whether all columns containing NaN are completely Nan
    any_is_all = torch.all(X_isnan.any(dim = 0) == X_isnan.all(dim = 0))
    print(f"All columns containing NaN are completely Nan: {any_is_all}")

    #Check which columns are NaN
    all_nan = X_isnan.all(dim = 0)
    nan_indices = torch.where(all_nan)[0]

    col_map_dict = default_X_col_map()
    all_nan_col_strings = [col_map_dict[ind] for ind in nan_indices.tolist()]
    
    print("Cols that are all NaN:\n", '\n'.join(all_nan_col_strings))

    with open('info_files/min_max_nan_cols.yaml', 'w') as file:
        yaml.dump(all_nan_col_strings, file)
   
    





"""
Investigate - Investigate Tensors
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def investigate_tensor():

    tensor_one_mapping = True
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    #X_data_tensor_name = 'X_data_tensor.pt'
    X_data_tensor_name = 'X_data_tensor_normalised.pt'
    #y_data_tensor_name = 'y_data_tensor.pt'

    X_data: Tensor = torch.load(tensor_dir / X_data_tensor_name)

    identify_min_max_NaN_source(X_data)



"""
Investigate - Investigate Mapping
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def investigate_index_mapping():

    tensor_one_mapping = True
    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    X_data_tensor_name = 'X_data_tensor.pt'
    #X_data_tensor_name = 'X_data_tensor_normalised.pt'
    y_data_tensor_name = 'y_data_tensor.pt'

    X_data: Tensor = torch.load(tensor_dir / X_data_tensor_name)
    y_data: Tensor = torch.load(tensor_dir / y_data_tensor_name)

    max_ind = len(X_data)
    missing_inds_X = [i for i in range(max_ind) if i not in X_data[:,0].tolist()]
    print(
        f'=================================================\n'
        f'Size X Tensor:\n{X_data.shape}\n'
        f'-------------------------------------------------\n'
        f'Missing indices X:\n{missing_inds_X}\n'
        f'=================================================\n'
    )




"""
Investigate - Investigate ID match
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def investigate_id_match():

    data_dir = Path("./data")
    tensor_dir = data_dir / "raw"

    X_data_max = 'apc_dataset_max.csv'
    X_data_key = 'apc_dataset_key.csv'
    y_data = 'mean_MR.csv'

    X_data_max_df = pd.read_csv(tensor_dir / X_data_max, low_memory = False)
    X_data_key_df = pd.read_csv(tensor_dir / X_data_key, low_memory = False)
    y_data_df = pd.read_csv(tensor_dir / y_data, low_memory = False)

    max_ids = set(X_data_max_df['WAFER_ID'].tolist())
    key_ids = set(X_data_key_df['WAFER_ID'].tolist())
    y_ids = set(y_data_df['WAFER_ID'].tolist())

    print('Max and Key ids match', max_ids == key_ids)

    y_not_in_apc = [id for id in y_ids if id not in max_ids | key_ids]
    y_not_in_max = [id for id in y_ids if id not in max_ids]
    y_not_in_key = [id for id in y_ids if id not in key_ids]

    print(
        f'Are y-data ids not in apc?\n'
        f'=================================================\n'
        f'joint apc:\n{y_not_in_apc}\n'
        f'-------------------------------------------------\n'
        f'apc max:\n{y_not_in_max}\n'
        f'-------------------------------------------------\n'
        f'apc key:\n{y_not_in_key}\n'
        f'=================================================\n'
    )

