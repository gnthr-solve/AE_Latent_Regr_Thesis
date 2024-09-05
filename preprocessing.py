
import torch
import pandas as pd
import numpy as np
import yaml
import json

from torch.utils.data import Dataset, DataLoader
from itertools import product
from pathlib import Path

"""
Preprocessing
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def eliminate_rows_with_NaN(df: pd.DataFrame):

    isna_any_mask = df.isna().any(axis = 1)

    df = df[isna_any_mask]

    return df