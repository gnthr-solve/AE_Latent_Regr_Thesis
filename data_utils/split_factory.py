
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

from .alignment import Alignment
from .datasets import TensorDataset

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


    def create_splits(self) -> dict[str, Subset]:

        train_size = int(self.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_indices, test_indices = random_split(range(len(self.dataset)), [train_size, test_size])
        
        y_data = self.dataset.y_data
        y_isnan = y_data[:, 1:].isnan().all(dim = 1)
        
        unlabeled_indices = torch.where(y_isnan)[0].tolist()
        labeled_indices = torch.where(~y_isnan)[0].tolist()
        

        splits = {
            'train_labeled': self._subset(list(set(train_indices).intersection(labeled_indices))),
            'train_unlabeled': self._subset(list(set(train_indices).intersection(unlabeled_indices))),
            'test_labeled': self._subset(list(set(test_indices).intersection(labeled_indices))),
            'test_unlabeled': self._subset(list(set(test_indices).intersection(unlabeled_indices)))
        }

        return splits


    def _subset(self, indices: list[int]) -> Subset:

        return Subset(dataset = self.dataset, indices = indices)




