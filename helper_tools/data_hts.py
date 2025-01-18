
import pandas as pd
import numpy as np
import json

from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field


"""
Data Helper Tools - Map Loader
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class MapLoader:
    """
    Helper class to load and transform the column and index mappings from json files.
    """
    def __call__(self, map_path: Path) -> dict[int, str]:

        with open(map_path, 'r') as f:
            map_dict: dict[str, str] = json.load(f)

        map_dict = {
            int(k): v 
            for k, v in map_dict.items() 
            if k.isdigit()
        }

        return map_dict


map_loader = MapLoader()




