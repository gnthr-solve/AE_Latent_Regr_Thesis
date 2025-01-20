
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset

from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from .alignment import Alignment
from .info import identifier_col

"""
DataSets - TimeSeriesDataset
-------------------------------------------------------------------------------------------------------------------------------------------
Dataset implemented for potential usage with Transformer models.
NOTE: Not tested & requires custom collate function to work with pytorch dataloaders.
"""
class TimeSeriesDataset(Dataset):

    def __init__(self, alignment: Alignment):

        self.alignment = alignment

        data_dir = Path.cwd().parent / 'data/timeseries_dataset'
        self.ts_dir = data_dir / 'timeseries'

        self.metadata_df = pd.read_csv(data_dir / 'metadata.csv', low_memory = False)
        # Arrange metadata_df rows to match index_map order
        index_order = [self.alignment.index_map[i] for i in sorted(self.alignment.index_map.keys())]
        self.metadata_df.set_index(identifier_col, inplace=True)
        self.metadata_df = self.metadata_df.loc[index_order].reset_index()


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata_df)


    def __getitem__(self, ndx):
        
        ndx_id = self.metadata_df[identifier_col].iat[ndx]
        
        id_ts_df = pd.read_parquet(self.ts_dir / f'{ndx_id}.parquet')

        return torch.tensor(data = id_ts_df.to_numpy(), dtype = torch.float32)




"""
DataSets - TensorDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class TensorDataset(Dataset):
    """
    Main Dataset subclass used in this work.
    Instances contain a full dataset, including the X_data (input), y_data (target, output),
    metadata and the alignment mappings.

    Attributes
    ----------
        alignment: Alignment
            Alignment linking tensor dimensions to their identity and meaning.
        metadata_df: pd.DataFrame
            Dataframe of metadata associated with each sample.
            Connected to the tensor entries via the 'mapping_idx'.
        X_data: Tensor
            Input data for the models.
        y_data: Tensor
            Output/target labels.
        X_dim: int
            Number of input features.
        y_dim: int
            Number of output features.

    """
    def __init__(self, X_data: Tensor, y_data: Tensor, metadata_df: pd.DataFrame, alignment: Alignment):

        self.alignm = alignment

        self.metadata_df = metadata_df

        self.X_data = X_data
        self.y_data = y_data

        self.X_dim = self.X_data.shape[-1]
        self.y_dim = self.y_data.shape[-1]


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X_data)


    def __getitem__(self, ndx):
        """
        Retrieves a data sample.

        Args:
            ndx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - X_data (torch.Tensor): The tensor representation of the input of the data sample.
                - y_data (torch.Tensor): The tensor representation of the label of the data sample.
        """

        return self.X_data[ndx], self.y_data[ndx]




