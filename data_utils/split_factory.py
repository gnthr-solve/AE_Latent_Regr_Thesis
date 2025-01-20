
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
    """
    Produces Subset instances of a TensorDataset by conducting a train-test-split 
    and separating the split data into labelled und unlabelled Subsets respectively.
    """
    def __init__(self, dataset: Dataset | TensorDataset, train_size = 0.8):

        self.dataset = dataset
        self.train_size = train_size

        self._create_splits()


    def _create_splits(self) -> dict[str, Subset]:

        train_size = int(self.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        train_indices, test_indices = random_split(range(len(self.dataset)), [train_size, test_size])
        
        y_data = self.dataset.y_data
        y_isnan = y_data[:, 1:].isnan().all(dim = 1)
        
        unlabeled_indices = torch.where(y_isnan)[0].tolist()
        labeled_indices = torch.where(~y_isnan)[0].tolist()
        

        self.splits = {
            'train':{
                'labelled': self._subset(list(set(train_indices).intersection(labeled_indices))),
                'unlabelled': self._subset(list(set(train_indices).intersection(unlabeled_indices))),
            },
            'test':{
                'labelled': self._subset(list(set(test_indices).intersection(labeled_indices))),
                'unlabelled': self._subset(list(set(test_indices).intersection(unlabeled_indices)))
            }
        }


    def retrieve(self, kind: str, combine: bool = False) -> dict[str, Subset] | Subset:
        """
        Retrieves the train or test data Subsets. 
        Optional 'combine' parameter allows returning the labelled and unlabelled Subsets as one Subset.

        Parameters
        ----------
            kind: str
                'test' or 'train'.
            combine: bool = False
                Combines labelled and unlabelled samples if True. Defaults to False.
        """
        subsets = self.splits[kind]

        if combine:
            indices = []

            for subset in subsets.values():
                indices += subset.indices

            return self._subset(indices)
        
        return subsets


    def _subset(self, indices: list[int]) -> Subset:

        return Subset(dataset = self.dataset, indices = indices)



"""

def unite_subsets(self, subset_names: list[str]) -> Subset:
        
        combined_indices = set()
        for name in subset_names:
            subset = self.splits[name]
            combined_indices.update(subset.indices)
        return Subset(self.dataset, list(combined_indices))

    def get_training_data(self, mode: str = 'full') -> Subset:
        
        if mode == 'full':
            return self.unite_subsets(['train_labeled', 'train_unlabeled'])
        elif mode == 'labeled':
            return self.splits['train_labeled']
        elif mode == 'unlabeled':
            return self.splits['train_unlabeled']
        else:
            raise ValueError(f"Unknown mode: {mode}")


    def _get_label_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        y_isnan = self.dataset.y_data[:, 1:].isnan().all(dim=1)
        return ~y_isnan, y_isnan
"""







