
import datetime as dt
import pandas as pd
import hashlib
import re

from collections import namedtuple
from pathlib import Path
from typing import Sequence

"""
Pandas Helper Tools - DataFrame Binning for Grouping
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def bin_df_by(df: pd.DataFrame, column: str, bins: int | Sequence):
        
    df_binned = df.copy()
    df_binned[f'{column}_bin'] = pd.cut(df_binned[column], bins)

    return df_binned



def analyze_by_bins(df: pd.DataFrame, column: str, bins: int, agg_columns: list[str]):

    df_binned = df.copy()
    df_binned[f'{column}_bin'] = pd.cut(df_binned[column], bins)
    
    result = df_binned.groupby(f'{column}_bin')[agg_columns].agg(['mean', 'std', 'count'])
    
    return result