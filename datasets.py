
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from helper_tools import default_index_map, default_X_col_map, default_y_col_map
from preprocessing.info import process_metadata_cols, identifier_col

"""
DataSets - Alignment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
@dataclass(slots=True)
class Alignment:

    index_map: dict[int, str] = field(default_factory = default_index_map)

    X_col_map: dict[int, str] = field(default_factory = default_X_col_map)
    y_col_map: dict[int, str] = field(default_factory = default_y_col_map)

    X_cols: list[str] = field(init=False)
    y_cols: list[str] = field(init=False)

    def __post_init__(self):

        self.X_cols = list(self.X_col_map.values())
        self.y_cols = list(self.y_col_map.values())




"""
DataSets - DataFrameDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class DataFrameDataset(Dataset):

    def __init__(self, joint_data_df: pd.DataFrame, alignment: Alignment = Alignment()):

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

    def __init__(self, X_data: Tensor, y_data: Tensor, metadata_df: pd.DataFrame, alignment: Alignment = Alignment()):

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
Datasets - Idea - SubsetFactory
-------------------------------------------------------------------------------------------------------------------------------------------
Subset from torch.utils.data allows to train or test on subset of dataset.
Metadata gives a lot of options to create conditional subsets.

Idea 1:
    Create a factory class that takes conditions on the metadata dataframe 
    and selects the indices where conditions hold true. 
    Then return corresponding subset of Dataset

Idea 2:
    Create a factory class that produces subsets for different model compositions.
    E.g. Sequential composition (i.e. Train Autoencoder first, then regression)
    could filter for indices where y_data is NaN for autoencoder training,
    then subset where it is not NaN for joint/End-To-End training
"""





"""
Tests
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    ###--- Test Alignment ---###
    
    alignment = Alignment()
    print(alignment.index_map)
    print(alignment.X_col_map)
    print(alignment.y_col_map)

    pass