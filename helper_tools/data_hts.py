
import torch
import pandas as pd
import numpy as np
import json

from torch import Tensor
from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser

from datasets import Alignment, TensorDataset

"""
Data Helper Tools - Default Factories
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MapLoader:

    def __call__(self, map_path: Path) -> dict[int, str]:

        with open(map_path, 'r') as f:
            map_dict: dict[str, str] = json.load(f)

        map_dict = {
            int(k): v 
            for k, v in map_dict.items() 
            if k.isdigit()
        }

        return map_dict



tensor_one_maps = True
tensor_one_infix = '_tensor' if tensor_one_maps else ''

map_loader = MapLoader()
index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
y_col_map = map_loader(Path(f'data/alignment_info/y{tensor_one_infix}_col_map.json'))

X_col_map_key = map_loader(Path(f'data/alignment_info/X_key{tensor_one_infix}_col_map.json'))
X_col_map_max = map_loader(Path(f'data/alignment_info/X_max{tensor_one_infix}_col_map.json'))



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

    contained_ids = [int(idx) for idx in indices]
    mask = metadata_df['mapping_idx'].isin(contained_ids)

    return metadata_df[mask]


"""
Data Helper Tools - Dataset Builder
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class DatasetBuilder:

    def __init__(
        self,
        kind: str = 'key', 
        normaliser: MinMaxNormaliser | ZScoreNormaliser = None,
        exclude_columns: list[str] = []
        ):

        data_dir = Path(f"./data")
        tensor_dir = data_dir / "tensors"

        self.exclude_columns = exclude_columns
        self.normaliser = normaliser

        self.index_map = index_map
        self.X_col_map = X_col_map_key if kind == 'key' else X_col_map_max
        self.y_col_map = y_col_map

        self.metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)

        self.X_data: torch.Tensor = torch.load(f = tensor_dir / f'X_data_{kind}_tensor.pt', weights_only = True)
        self.y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt', weights_only = True)

        
    def exclude_columns_and_update_mapping(self):

        keys_to_keep = [key for key in self.X_col_map.values() if key not in self.exclude_columns]
        indices_to_keep = [int(k) for k, v in self.X_col_map.items() if v in keys_to_keep]

        self.X_data = self.X_data[:, indices_to_keep]
        self.X_col_map = {i: v for i, v in enumerate(keys_to_keep)}


    def normalise(self):

        with torch.no_grad():
            self.X_data[:, 1:] = self.normaliser.normalise(self.X_data[:, 1:])
            
            X_data_isnan = self.X_data.isnan().all(dim = 0)
            nan_indices = torch.where(X_data_isnan)[0].tolist()
            
            self.X_data = self.X_data[:, ~X_data_isnan]

        filtered_col_map = {idx: val for idx, val in self.X_col_map.items() if idx not in nan_indices}
        self.X_col_map = {i: v for i, v in enumerate(filtered_col_map.values())}


    def build_dataset(self) -> TensorDataset:

        if self.exclude_columns:
            self.exclude_columns_and_update_mapping()

        if self.normaliser is not None:
            self.normalise()

        alignment = Alignment(index_map = self.index_map, X_col_map = self.X_col_map, y_col_map = self.y_col_map)

        dataset = TensorDataset(self.X_data, self.y_data, self.metadata_df, alignment = alignment)

        return dataset
    



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
