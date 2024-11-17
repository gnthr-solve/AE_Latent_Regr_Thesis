
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
DataSets - Alignment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
@dataclass(slots=True)
class AlignmentPrev:

    index_map: dict[int, str] = field(default_factory = {})

    X_col_map: dict[int, str] = field(default_factory = {})
    y_col_map: dict[int, str] = field(default_factory = {})

    X_cols: list[str] = field(init=False)
    y_cols: list[str] = field(init=False)

    def __post_init__(self):

        self.X_cols = list(self.X_col_map.values())
        self.y_cols = list(self.y_col_map.values())




"""
DataSets - Alignment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Alignment:

    def __init__(self, index_map: dict[int, str], X_col_map: dict[int, str], y_col_map: dict[int, str]):

        self.index_map = index_map

        self.X_col_map = X_col_map
        self.y_col_map = y_col_map

        
    def filter_col_map(self, filter_out_values: Sequence[str]| Sequence[int], by_key: bool):
    
            if by_key:
                filtered_col_map = {idx: key for idx, key in self.X_col_map.items() if key not in filter_out_values}
                
            else:
                filtered_col_map = {idx: key for idx, key in self.X_col_map.items() if idx not in filter_out_values}
    
            self.X_col_map = {idx: key for idx, key in enumerate(filtered_col_map.values())}


    def retrieve_col_labels(self, indices: Sequence[int], from_X: bool = True) -> list[str]:
        
        if from_X:
            return [self.X_col_map[idx] for idx in indices]
        else:
            return [self.y_col_map[idx] for idx in indices]
    

"""
DataSets - Full Alignments
-------------------------------------------------------------------------------------------------------------------------------------------
"""
tensor_one_maps = True
tensor_one_infix = '_tensor' if tensor_one_maps else ''

index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
y_col_map = map_loader(Path(f'data/alignment_info/y{tensor_one_infix}_col_map.json'))

X_col_map_key = map_loader(Path(f'data/alignment_info/X_key{tensor_one_infix}_col_map.json'))
X_col_map_max = map_loader(Path(f'data/alignment_info/X_max{tensor_one_infix}_col_map.json'))

# Create Alignments
alignment_key = Alignment(index_map = index_map, X_col_map = X_col_map_key, y_col_map = y_col_map)
alignment_max = Alignment(index_map = index_map, X_col_map = X_col_map_max, y_col_map = y_col_map)

