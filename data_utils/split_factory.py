
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




class SplitSubsetFactory:
    def __init__(self, 
                 dataset: TensorDataset, 
                 train_size: float = 0.8,
                 random_state: int | None = None):
        self.dataset = dataset
        self.train_size = train_size
        self.random_state = random_state
        
    def _get_label_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute boolean masks for labeled/unlabeled data."""
        y_isnan = self.dataset.y_data[:, 1:].isnan().all(dim=1)
        return ~y_isnan, y_isnan
    
    def _stratified_split(self, indices: list[int], label_mask: torch.Tensor) -> tuple[list[int], list[int]]:
        """Perform stratified split maintaining label distribution."""
        labeled_indices = [idx for idx in indices if label_mask[idx]]
        unlabeled_indices = [idx for idx in indices if not label_mask[idx]]
        
        train_labeled = labeled_indices[:int(len(labeled_indices) * self.train_size)]
        test_labeled = labeled_indices[int(len(labeled_indices) * self.train_size):]
        
        return train_labeled, test_labeled

    def create_splits(self, stratify: bool = True) -> dict[str, Subset]:
        """
        Create dataset splits with optional stratification.
        
        Args:
            stratify: If True, maintains label/unlabeled ratio in splits
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            
        labeled_mask, unlabeled_mask = self._get_label_masks()
        all_indices = range(len(self.dataset))
        
        if stratify:
            train_labeled, test_labeled = self._stratified_split(
                list(range(len(self.dataset))), 
                labeled_mask
            )
            train_unlabeled, test_unlabeled = self._stratified_split(
                list(range(len(self.dataset))), 
                unlabeled_mask
            )
        else:
            # Original random split logic
            train_size = int(self.train_size * len(self.dataset))
            train_indices, test_indices = random_split(all_indices, 
                [train_size, len(self.dataset) - train_size])
            
            labeled_indices = torch.where(labeled_mask)[0].tolist()
            unlabeled_indices = torch.where(unlabeled_mask)[0].tolist()
            
            train_labeled = list(set(train_indices).intersection(labeled_indices))
            train_unlabeled = list(set(train_indices).intersection(unlabeled_indices))
            test_labeled = list(set(test_indices).intersection(labeled_indices))
            test_unlabeled = list(set(test_indices).intersection(unlabeled_indices))
        
        return {
            'train_labeled': Subset(self.dataset, train_labeled),
            'train_unlabeled': Subset(self.dataset, train_unlabeled),
            'test_labeled': Subset(self.dataset, test_labeled),
            'test_unlabeled': Subset(self.dataset, test_unlabeled)
        }
    
    def get_split_statistics(self, splits: dict[str, Subset]) -> dict[str, int]:
        """Return statistics about the splits."""
        return {
            name: len(subset) for name, subset in splits.items()
        }