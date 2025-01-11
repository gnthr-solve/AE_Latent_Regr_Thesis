
import datetime as dt
import pandas as pd
import numpy as np
import hashlib
import re

from collections import namedtuple
from pathlib import Path
from typing import Sequence, Callable, Any
from itertools import product
from functools import wraps



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



"""
Pandas Helper Tools - Outlier Filter
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def remove_outliers_by_zscore(
        df: pd.DataFrame, 
        column: str,
        z_threshold: int
    ) -> pd.DataFrame:
    """
    Remove outliers from DataFrame based on specified column.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        z_threshold: multiple of standard deviation within tolerance
        
    Returns:
        DataFrame with outliers removed
    """
    
    values = df[column].values
    
    mean_value = np.mean(values)
    std_dev = np.std(values)

    lower_threshold = mean_value - z_threshold * std_dev
    upper_threshold = mean_value + z_threshold * std_dev
    
    outlier_mask = (values < lower_threshold) | (values > upper_threshold)

    return df[~outlier_mask].copy()



def remove_outliers_by_mad(
        df: pd.DataFrame, 
        column: str,
        threshold: int
    ) -> pd.DataFrame:
    
    values = df[column].values
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    outlier_mask = np.abs(values - median) > threshold * mad
    print(
        f'Removed {len(df[outlier_mask])} outlier rows.'
    )
    return df[~outlier_mask].copy()



def bound_by_col(
        df: pd.DataFrame, 
        column: str,
        upper: float,
        lower: float = None,
    ) -> pd.DataFrame:

    values = df[column].values

    if lower is None:
        mask = values < upper
    else:
        mask = (values > lower) & (values < upper)

    return df[mask].copy()