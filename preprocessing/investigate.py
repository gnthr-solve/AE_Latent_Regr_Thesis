

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
