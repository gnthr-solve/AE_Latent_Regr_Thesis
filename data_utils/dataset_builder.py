
import torch
import pandas as pd
import numpy as np
import json

from torch import Tensor
from itertools import product
from pathlib import Path
from typing import Callable

from preprocessing.normalisers import MinMaxNormaliser, ZScoreNormaliser
from helper_tools import constant_mask

from .datasets import TensorDataset
from .alignment import Alignment, load_init_alignment
from .info import time_col

"""
Data - Dataset Builder
-------------------------------------------------------------------------------------------------------------------------------------------
Setup class following the Builder design pattern.
"""
class DatasetBuilder:
    """
    Builder class that assembles a TensorDataset instance by loading and adjusting 
    the data tensors and mappings.

    Input Parameters
    ----------
        kind: str = 'key'
            Aggregated dataset kind to use, can be 'key' or 'max', defaults to 'key'
        normaliser: MinMaxNormaliser | ZScoreNormaliser = None,
            Normaliser class to normalise the X/input tensor with.
            If None, no normalisation is performed.
        exclude_columns: list[str] = []
            List of strings of input data columns/features to exclude from the data.
            Allows using a reduced dataset for experiments.

    """
    def __init__(
        self,
        kind: str = 'key', 
        normaliser: MinMaxNormaliser | ZScoreNormaliser = None,
        exclude_columns: list[str] = [],
        filter_condition: Callable[[pd.DataFrame], pd.Series] = None,
        exclude_const_columns: bool = True,
        ):

        data_dir = Path(f"./data")
        tensor_dir = data_dir / "tensors"

        self.normaliser = normaliser
        self.exclude_columns = exclude_columns
        self.filter_condition = filter_condition
        self.exclude_const_columns = exclude_const_columns

        self.alignment = load_init_alignment(kind = kind)

        self.metadata_df = pd.read_csv(data_dir / "metadata.csv", low_memory = False)
        self.metadata_df.loc[:, time_col] = pd.to_datetime(self.metadata_df[time_col], format=r'%Y-%m-%d %H:%M:%S')

        self.X_data: torch.Tensor = torch.load(f = tensor_dir / f'X_data_{kind}_tensor.pt', weights_only = True)
        self.y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt', weights_only = True)

        
    def exclude_columns_and_update_mapping(self):
        """
        Uses the exclude_columns attribute to remove features from the input tensor and adjust the alignment mapping.
        """
        indices_to_keep = [int(idx) for idx, key in self.alignment.X_col_map.items() if key not in self.exclude_columns]

        self.X_data = self.X_data[:, indices_to_keep]
        self.alignment.filter_col_map(filter_out_values = self.exclude_columns, by_key = True)


    def filter_rows_by_metadata(self):
        """
        Method to use a metadata-filter-condition to remove rows/samples from the data.
        For now only a dummy condition based on the time stamp is used.

        Works by retrieving the mapping_idx column of the segment of the metadata dataframe, 
        that satisfies the condition, and creating a mask of tensor segments to keep.
        """
        row_mask = self.filter_condition(self.metadata_df)
        #print(row_mask)
        self.metadata_df = self.metadata_df[row_mask]
        indices = self.metadata_df['mapping_idx'].tolist()

        # Convert indices to a tensor
        indices_tensor = torch.tensor(indices, dtype=self.X_data.dtype)

        # Unsqueezing allows for broadcasting the comparison operation which then results in a matrix mask tensor, reduced by any
        data_mask = (self.X_data[:, 0].unsqueeze(-1) == indices_tensor).any(dim=-1)

        self.X_data = self.X_data[data_mask]
        self.y_data = self.y_data[data_mask]


    def remove_constant_features(self):
        """
        Removes constant columns and then updates alignment.
        """
        with torch.no_grad():

            X_data_const_mask = constant_mask(tensor = self.X_data, axis = 0)
            const_indices = torch.where(X_data_const_mask)[0].tolist()

            self.X_data = self.X_data[:, ~X_data_const_mask]
            
        self.alignment.filter_col_map(filter_out_values = const_indices, by_key = False)


    def normalise(self):
        """
        Normalises the input data, using the normaliser attribute, first, then updates alignment.
        """
        with torch.no_grad():

            self.X_data[:, 1:] = self.normaliser.normalise(self.X_data[:, 1:])
            
            X_data_isnan = self.X_data.isnan().all(dim = 0)
            nan_indices = torch.where(X_data_isnan)[0].tolist()
            
            self.X_data = self.X_data[:, ~X_data_isnan]

        self.alignment.filter_col_map(filter_out_values = nan_indices, by_key = False)


    def build_dataset(self) -> TensorDataset:
        """
        Principal builder method.
        Applies normalisation and filter operations if applicable, then instantiates and returns
        the TensorDataset.
        """
        if self.filter_condition is not None:
            self.filter_rows_by_metadata()

        if self.exclude_columns:
            self.exclude_columns_and_update_mapping()

        if self.exclude_const_columns:
            self.remove_constant_features()

        if self.normaliser is not None:
            self.normalise()

        
        dataset = TensorDataset(self.X_data, self.y_data, self.metadata_df, alignment = self.alignment)

        return dataset
    


