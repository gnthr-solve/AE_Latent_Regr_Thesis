
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

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

class SplitSubsetFactory:

    def __init__(self, dataset: Dataset | TensorDataset, train_size = 0.8):

        self.dataset = dataset
        self.train_size = train_size


    def create_splits_alpha(self):

        train_size = int(self.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_indices, test_indices = random_split(range(len(self.dataset)), [train_size, test_size])
        print(
            f'=================================================\n'
            f'Type train indices: \n{type(train_indices)}\n'
            f'Length train indices: \n{len(train_indices)}\n'
            f'Train indices: \n{train_indices}\n'
            f'-------------------------------------------------\n'
            f'Type test indices: \n{type(test_indices)}\n'
            f'Length test indices: \n{len(test_indices)}\n'
            f'Test indices: \n{test_indices}\n'
            f'=================================================\n'
        )


        labeled_indices = [i for i, y in enumerate(self.dataset.y_data) if not torch.isnan(y).any()]
        unlabeled_indices = [i for i in range(len(self.dataset)) if i not in labeled_indices]

        splits = {
            'train_labeled': list(set(train_indices).intersection(labeled_indices)),
            'train_unlabeled': list(set(train_indices).intersection(unlabeled_indices)),
            'test_labeled': list(set(test_indices).intersection(labeled_indices)),
            'test_unlabeled': list(set(test_indices).intersection(unlabeled_indices))
        }

        return splits
    


    def create_splits(self) -> dict[str, Subset]:

        train_size = int(self.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_indices, test_indices = random_split(range(len(self.dataset)), [train_size, test_size])
        print(
            f'=================================================\n'
            f'Type train indices: \n{type(train_indices)}\n'
            f'Length train indices: \n{len(train_indices)}\n'
            f'Train indices: \n{train_indices}\n'
            f'-------------------------------------------------\n'
            f'Type test indices: \n{type(test_indices)}\n'
            f'Length test indices: \n{len(test_indices)}\n'
            f'Test indices: \n{test_indices}\n'
            f'=================================================\n'
        )

        y_data = self.dataset.y_data
        y_isnan = y_data[:, 1:].isnan().all(dim = 1)
        unlabeled_indices = torch.tensor(y_data[y_isnan, 0], dtype = torch.int32)
        
        print(
            f'=================================================\n'
            f'Shape y_data: \n{y_data.shape}\n'
            f'y_data[:10]: \n{y_data[:10]}\n'
            f'-------------------------------------------------\n'
            f'Shape y_isnan: \n{y_isnan.shape}\n'
            f'y_isnan[:10]: \n{y_isnan[:10]}\n'
            f'-------------------------------------------------\n'
            f'Shape unlabeled_indices: \n{unlabeled_indices.shape}\n'
            f'unlabeled_indices[:10]: \n{unlabeled_indices[:10]}\n'
            f'=================================================\n'
        )

        unlabeled_indices = unlabeled_indices.tolist()
        labeled_indices = [i for i in range(len(self.dataset)) if i not in unlabeled_indices]

        splits = {
            'train_labeled': self._subset(list(set(train_indices).intersection(labeled_indices))),
            'train_unlabeled': self._subset(list(set(train_indices).intersection(unlabeled_indices))),
            'test_labeled': self._subset(list(set(test_indices).intersection(labeled_indices))),
            'test_unlabeled': self._subset(list(set(test_indices).intersection(unlabeled_indices)))
        }

        return splits


    def _subset(self, indices: list[int]) -> Subset:

        return Subset(dataset = self.dataset, indices = indices)




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