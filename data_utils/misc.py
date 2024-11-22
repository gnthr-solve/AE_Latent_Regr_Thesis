
import torch
import pandas as pd
import numpy as np
import json

from torch import Tensor
from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from .datasets import TensorDataset
from .alignment import Alignment

"""
Data Helper Tools - Load Tensor segments
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def load_tensor_segment(tensor_path: Path, indices: list[int]) -> Tensor:

    tensor = torch.load(tensor_path, weights_only = True)
    tensor_segment = tensor[indices]

    return tensor_segment



"""
Data Helper Tools - tensor to df
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def X_tensor_to_df(X_data: Tensor, alignment, metadata: pd.DataFrame) -> pd.DataFrame:

    index_map = alignment.index_map
    col_map = alignment.X_col_map

    indices = X_data[:, 0].tolist()
    df_dict = {'WAFER_ID': [index_map[int(idx)] for idx in indices]}
    df_dict.update({col_map[i]: X_data[:, i].tolist() for i in range(1, X_data.shape[1])})

    df = pd.DataFrame(df_dict)
    df = df.merge(metadata, on = 'WAFER_ID')

    return df
    


def retrieve_metadata(indices, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a Sequence of indices by a Subset of the Dataset, retrieve the corresponding metadata.
    """
    contained_ids = [int(idx) for idx in indices]
    mask = metadata_df['mapping_idx'].isin(contained_ids)

    return metadata_df[mask]



"""
Data Helper Tools - Storage
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def remove_columns_and_update_mapping(tensor: Tensor, mapping: dict[int, str], keys_to_remove: list[str]):

    keys_to_keep = [key for key in mapping.values() if key not in keys_to_remove]
    indices_to_keep = [int(k) for k, v in mapping.items() if v in keys_to_keep]

    new_tensor = tensor[:, indices_to_keep]
    new_mapping = {i: v for i, v in enumerate(keys_to_keep)}

    return new_tensor, new_mapping