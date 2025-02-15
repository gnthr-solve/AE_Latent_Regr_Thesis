
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split

from itertools import product
from pathlib import Path
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass, field

from helper_tools.data_hts import map_loader

"""
DataSets - Alignment for TimeSeriesDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""
@dataclass(slots=True)
class AlignmentTS:

    index_map: dict[int, str] = field(default_factory = {})

    




"""
DataSets - TensorDataset Alignment
-------------------------------------------------------------------------------------------------------------------------------------------
Mapping container class incorporated into TensorDataset.
"""
class Alignment:
    """
    Container class relating tensor indices to row-identity 
    and column-name (i.e. feature meaning) of the original dataframe format.

    The tensors of the data referred to here have shape (m,n+1), where m is the number of samples
    and n the number of features. 
    First entry of every sample, i.e. tensor[:, 0], represents the mapping index, 
    that relates the tensor entry to its identity.


    Attributes
    ----------
        index_map: dict[int, str]
            Dictionary mapping the first entry of a sample to its real ID.
        X_col_map: dict[int, str]
            Dictionary mapping the second dimension of input data tensors to feature names.
        y_col_map: dict[int, str]
            Dictionary mapping the second dimension of target-label data tensors to feature names.

    """
    def __init__(self, index_map: dict[int, str], X_col_map: dict[int, str], y_col_map: dict[int, str]):

        self.index_map = index_map

        self.X_col_map = X_col_map
        self.y_col_map = y_col_map

        
    def filter_col_map(self, filter_out_values: Sequence[str]| Sequence[int], by_key: bool):
        """
        Remove input features from the alignment X_col_map.
        Allows preserving the alignment if features were removed, e.g. by normalisation.

        Parameters
        ----------
            filter_out_values: Sequence[str]| Sequence[int]
                Sequence of integers or strings to remove from the mapping.
                If by_key == True, elements of the sequence need to be strings, otherwise integers.
            by_key: bool
                If True removes features by name, if False by integer value
        """
        if by_key:
            filtered_col_map = {idx: key for idx, key in self.X_col_map.items() if key not in filter_out_values}
            
        else:
            filtered_col_map = {idx: key for idx, key in self.X_col_map.items() if idx not in filter_out_values}

        self.X_col_map = {idx: key for idx, key in enumerate(filtered_col_map.values())}


    def retrieve_col_labels(self, indices: Sequence[int], from_X: bool = True) -> list[str]:
        """
        Retrieves feature names given a list of tensor indices in the feature dimension.

        Parameters
        ----------
            indices: Sequence[int]
                Sequence of integers for which to retrieve the name from the mapping.
            from_X: bool = True
                Whether to retrieve names for input features (from_X == True) or target features (from_X == False).
                Defaults to input features.
        
        Returns:
            list[str]
                List of names for given indices, extracted from the mapping.
        """
        if from_X:
            return [self.X_col_map[idx] for idx in indices]
        else:
            return [self.y_col_map[idx] for idx in indices]
    



"""
DataSets - Load Maps and Initialise Alignment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def load_init_alignment(kind: str = 'key'):
    """
    Load the index and column maps and instantiate an Alignment with them.
    Index map and y-col map coincide between key and max, due to preprocessing.

    Parameters
    ----------
        kind: str = 'key'
            Name of aggregate dataset, defaults to key.

    Returns:
        Alignment
            Alignment instance of chose dataset.
    """
    index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
    y_col_map = map_loader(Path(f'data/alignment_info/y_tensor_col_map.json'))

    X_col_map = map_loader(Path(f'data/alignment_info/X_{kind}_tensor_col_map.json'))

    # Create Alignments
    alignment = Alignment(index_map = index_map, X_col_map = X_col_map, y_col_map = y_col_map)
    
    return alignment


# index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
# y_col_map = map_loader(Path(f'data/alignment_info/y_tensor_col_map.json'))

# X_col_map_key = map_loader(Path(f'data/alignment_info/X_key_tensor_col_map.json'))
# X_col_map_max = map_loader(Path(f'data/alignment_info/X_max_tensor_col_map.json'))

# # Create Alignments
# alignment_key = Alignment(index_map = index_map, X_col_map = X_col_map_key, y_col_map = y_col_map)
# alignment_max = Alignment(index_map = index_map, X_col_map = X_col_map_max, y_col_map = y_col_map)

