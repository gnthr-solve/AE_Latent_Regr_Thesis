
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from .alignment import Alignment


"""
DataSets - DataFrameDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class DataFrameDataset(Dataset):

    def __init__(self, joint_data_df: pd.DataFrame, alignment: Alignment):

        self.alignm = alignment
        self.data_df = joint_data_df

        X_data_df = self.data_df[self.alignm.X_cols]
        y_data_df = self.data_df[self.alignm.y_cols]

        self.metadata_df = self.data_df.drop(columns = self.alignm.X_cols + self.alignm.y_cols)

        self.X_data = torch.tensor(X_data_df.values, dtype=torch.float32)  
        self.y_data = torch.tensor(y_data_df.values, dtype=torch.float32)

        self.X_dim = self.X_data.shape[1:]
        self.y_dim = self.y_data.shape[1:]


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_df)


    def __getitem__(self, ndx):
        """
        Retrieves a data sample and its associated metadata.

        Args:
            ndx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - X_data (torch.Tensor): The tensor representation of the data sample.
                - y_data (torch.Tensor): The tensor representation of the data sample.
        """

        return self.X_data[ndx], self.y_data[ndx]




"""
DataSets - TensorDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class TensorDataset(Dataset):

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
                - X_data (torch.Tensor): The tensor representation of the data sample.
                - y_data (torch.Tensor): The tensor representation of the data sample.
        """

        return self.X_data[ndx], self.y_data[ndx]




