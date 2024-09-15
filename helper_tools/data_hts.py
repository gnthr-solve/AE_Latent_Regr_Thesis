
import torch
import pandas as pd
import numpy as np
import yaml
import json

from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field


"""
Data Helper Tools - Default Factories
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MapLoader:

    def __init__(self, map_path: Path):
        self.map_path = map_path

    def __call__(self):

        with open(self.map_path, 'r') as f:
            map_dict: dict[str, str] = json.load(f)

        map_dict = {
            int(k): v 
            for k, v in map_dict.items() 
            if k.isdigit()
        }

        return map_dict



tensor_one_maps = True
tensor_one_infix = '_tensor' if tensor_one_maps else ''

default_index_map = MapLoader(Path('data/alignment_info/index_id_map.json'))
default_X_col_map = MapLoader(Path(f'data/alignment_info/X{tensor_one_infix}_col_map.json'))
default_y_col_map = MapLoader(Path(f'data/alignment_info/y{tensor_one_infix}_col_map.json'))


