
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
    """
    Takes a dataframe, a column and a bins argument and creates a new column,
    assigning the rows to bins for analyses.
    
    Parameters
    ----------
        df: pd.DataFrame
            Input dataframe
        column: str
            Column that the dataframe is binned by.
        bins: int | Sequence
            Either an integer corresponding to the number of bins, or a sequence of custom bins.
    
    Returns:
        df_binned: pd.DataFrame 
            Input dataframe with additional column that represents bin membership.
    """
    df_binned = df.copy()
    df_binned[f'{column}_bin'] = pd.cut(df_binned[column], bins)

    return df_binned




"""
Pandas Helper Tools - Analyse dataframe by binning and aggregating.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Used primarily for refining hyperparameter search spaces based on previous results.
"""
def analyze_by_bins(
        df: pd.DataFrame, 
        column: str, 
        bins: int, 
        agg_columns: list[str],
        agg_funcs: list[str] | list[Callable] = ['mean', 'std', 'count'],
    ):
    """
    Bins a dataframe by values in 'column' using the 'bins' argument, 
    performs a groupby operation and aggregates the 'agg_columns' by bin membership.
    
    Parameters
    ----------
        df: pd.DataFrame
            Input dataframe
        column: str
            Column that the dataframe is binned by.
        bins: int | Sequence
            Either an integer corresponding to the number of bins, or a sequence of custom bins.
        agg_columns: list[str]
            Columns to be aggregated for the analysis.
        agg_funcs: list[str] | list[Callable]
            List of functions to use for the aggregation, based on pandas .agg signature.
            Defaults to 'mean', 'std' and 'count'.
    
    Returns:
        pd.DataFrame 
            Column-multi-indexed dataframe with top index in the 'agg_columns' 
            and aggregation func in the lower index.
    """
    df_binned = df.copy()
    df_binned[f'{column}_bin'] = pd.cut(df_binned[column], bins)
    
    result = df_binned.groupby(f'{column}_bin')[agg_columns].agg(agg_funcs)
    
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
    Remove outliers from DataFrame based on specified column and a Z-score threshold.
    
    Parameters
    ----------
        df: pd.DataFrame
            Input DataFrame
        column: str
            Column name to check for outliers
        z_threshold: int
            Multiple of standard deviation within tolerance
        
    Returns:
        pd.DataFrame 
            Dataframe with outliers outside of tolerance removed.
    """
    
    values = df[column].values
    
    mean_value = np.mean(values)
    std_dev = np.std(values)

    lower_threshold = mean_value - z_threshold * std_dev
    upper_threshold = mean_value + z_threshold * std_dev
    
    outlier_mask = (values < lower_threshold) | (values > upper_threshold)

    return df[~outlier_mask].copy()




"""
Pandas Helper Tools - Filter Outliers by MAD
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Removing outliers by MAD score proved more effective for result distributions, 
as they tend to be skewed towards zero and outliers have great impact on the mean, but not the median.
"""
def remove_outliers_by_mad(
        df: pd.DataFrame, 
        column: str,
        threshold: int
    ) -> pd.DataFrame:
    """
    Remove outliers from DataFrame based on multiple of MAD score.
    Converts column to numpy array, calculates the column-median 
    and the median of the absolute deviations from the column-median (MAD).
    Filters out rows that deviate more than threshold*MAD from the column-median.

    Parameters
    ----------
        df: pd.DataFrame
            Input DataFrame
        column: str
            Column name to check for outliers
        threshold: int
            Multiple of MAD score within tolerance
        
    Returns:
        pd.DataFrame 
            Dataframe with outliers outside of tolerance removed.
    """

    values = df[column].values
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    outlier_mask = np.abs(values - median) > threshold * mad
    
    return df[~outlier_mask].copy()




"""
Pandas Helper Tools - Filter Outliers by Bounds
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Used to filter outliers by manually selected bound.
"""
def bound_by_col(
        df: pd.DataFrame, 
        column: str,
        upper: float,
        lower: float = None,
    ) -> pd.DataFrame:
    """
    Remove segments from DataFrame based column values being within 'lower' and 'upper' bounds.

    Parameters
    ----------
        df: pd.DataFrame
            Input DataFrame.
        column: str
            Column name to bound by.
        upper: float
            Upper bound on acceptable values.
        lower: float = None
            Lower bound on acceptable values. 
            If not given as an argument will only use upper bound.

    Returns:
        pd.DataFrame 
            Dataframe with only rows whose values in 'column' are in the desired bounds.
    """
    values = df[column].values

    if lower is None:
        mask = values < upper
    else:
        mask = (values > lower) & (values < upper)

    return df[mask].copy()