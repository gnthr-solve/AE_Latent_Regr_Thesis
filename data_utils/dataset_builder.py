
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

from .datasets import TensorDataset
from .alignment import Alignment, alignment_key, alignment_max
from .info import time_col

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

        self.alignment = alignment_key if kind == 'key' else alignment_max

        self.metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)
        self.metadata_df.loc[:, time_col] = pd.to_datetime(self.metadata_df[time_col], format=r'%Y-%m-%d %H:%M:%S')

        self.X_data: torch.Tensor = torch.load(f = tensor_dir / f'X_data_{kind}_tensor.pt', weights_only = True)
        self.y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt', weights_only = True)

        
    def exclude_columns_and_update_mapping(self):

        indices_to_keep = [int(idx) for idx, key in self.alignment.X_col_map.items() if key not in self.exclude_columns]

        self.X_data = self.X_data[:, indices_to_keep]
        self.alignment.filter_col_map(filter_out_values = self.exclude_columns, by_key = True)


    def exclude_rows_by_metadata(self, filter_condition = None):

        #dummy filter condition for now
        filter_condition = lambda df: df[time_col] > pd.Timestamp('2023-08-01')

        row_mask = filter_condition(self.metadata_df)

        self.metadata_df = self.metadata_df[row_mask]
        indices = self.metadata_df['mapping_idx'].tolist()

        # Convert indices to a tensor
        indices_tensor = torch.tensor(indices, dtype=self.X_data.dtype)

        # Unsqueezing allows for broadcasting the comparison operation which then results in a matrix mask tensor, reduced by any
        data_mask = (self.X_data[:, 0].unsqueeze(-1) == indices_tensor).any(dim=-1)

        self.X_data = self.X_data[data_mask]
        self.y_data = self.y_data[data_mask]


    def normalise(self):

        with torch.no_grad():

            self.X_data[:, 1:] = self.normaliser.normalise(self.X_data[:, 1:])
            
            X_data_isnan = self.X_data.isnan().all(dim = 0)
            nan_indices = torch.where(X_data_isnan)[0].tolist()
            
            self.X_data = self.X_data[:, ~X_data_isnan]

        self.alignment.filter_col_map(filter_out_values = nan_indices, by_key = False)


    def build_dataset(self) -> TensorDataset:

        if self.exclude_columns:
            self.exclude_columns_and_update_mapping()

        if self.normaliser is not None:
            self.normalise()

        self.exclude_rows_by_metadata()

        dataset = TensorDataset(self.X_data, self.y_data, self.metadata_df, alignment = self.alignment)

        return dataset
    


