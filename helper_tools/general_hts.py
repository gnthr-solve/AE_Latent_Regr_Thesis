
import torch
import time
import numpy as np
import pandas as pd
import re

from torch import Tensor
from typing import Callable, Any
from itertools import product
from functools import wraps



"""
Timer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def simple_timer(func):
    """
    Simple timing decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        exec_time = end_time - start_time
        
        print(exec_time)
        
        return result
    
    return wrapper



"""
Miscellaneous
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def print_nested_dict(d: dict, indent: int = 0):
    
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))




def dict_str(d: dict):
    """
    Create a dictionary string representation that allows printing key value pairs line by line.
    
    Args:
        d: dict
            Input dictionary
    
    Returns:
        str
            String of of 'key: value' pairs separated by newline.
    """

    dict_strs = [
        f"{key}: {value}"
        for key, value in d.items()
    ]

    return ',\n'.join(dict_strs)




"""
Dictionaries
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def split_dict(
        d: dict, 
        condition: Callable[[Any], bool], 
        on_key: bool = False
    ) -> tuple[dict, dict]:
    """
    Split a dictionary into two based on a condition applied to either keys or values.
    
    Parameters
    ----------
        d: dict
            Input dictionary
        condition: Callable[[Any], bool]
            Boolean function to evaluate on either keys or values
        on_key: bool
            If True, apply condition to keys. If False, apply to values
    
    Returns:
        tuple[dict] 
            tuple of (dict_true, dict_false) where condition is satisfied/violated
    """
    dict_true = {}
    dict_false = {}
    
    for k, v in d.items():
        target = k if on_key else v
        if condition(target):
            dict_true[k] = v
        else:
            dict_false[k] = v
            
    return dict_true, dict_false



def map_dict_keys(d: dict[str, Any], key_map: dict[str, str]):
    """
    Replace dictionary keys with others for e.g. data obfuscation or plotting
    
    Parameters
    ----------
        d: dict
            Input dictionary
        key_map: dict[str, str]
            Dictionary of kind [old_key, new_key]
    
    Returns:
        dict
            Dictionary where old keys are replaced with values from key_map
    """
    return {key_map.get(k, k): v for k, v in d.items()}
    



"""
Normalise
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def normalise_dataframe(df: pd.DataFrame, columns: list[str]):
    """
    Normalise specified columns of a dataframe with min-max normalisation.
    
    Parameters
    ----------
        df: pd.DataFrame
            Input DataFrame.
        columns: list[str]
            List of columns to normalise.
    
    Returns:
        normalised_df: pd.DataFrame
            Input DataFrame where 'columns' are normalised to interval [0,1].
    """
    normalised_df = df.copy()
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
    
        normalised_df[col] = (df[col] - min_val) / (max_val - min_val)

    return normalised_df




def normalise_tensor(tensor: Tensor):
    """
    Normalise tensor min-max normalisation.
    
    Parameters
    ----------
        tensor: Tensor
            Input tensor of shape (m, n).
    
    Returns:
        normalised_tensor: Tensor
            Input tensor with dimension 1 normalised to interval [0,1].
    """
    min_val = tensor.min(dim = 0, keepdim = True)[0]
    max_val = tensor.max(dim = 0, keepdim = True)[0]
    
    normalised_tensor = (tensor - min_val) / (max_val - min_val)
    
    return normalised_tensor




"""
String Discriminator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class StringDiscriminator:
    """
    Callable meant to search a string for any substring in list of interest 
    and to be applied as a boolean check in comprehensions.
    Can be used to filter out strings (filter_out = True), i.e. will return True if none of the substrings are found.

    Parameters
    ----------
        list_of_interest: list[str]
            List of strings to search for.
        filter_out: bool
            Boolean flag to indicate whether to return a positive or a negative result for a match.
    """

    def __init__(self, list_of_interest: list[str], filter_out: bool = False):

        self.list_of_interest = list_of_interest
        self.filter_out = filter_out
    

    def __call__(self, string: str) -> bool:

        result = any(re.search(ref_string, string) for ref_string in self.list_of_interest)

        return self.filter_out ^ result
    



"""
Exceptions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AbortTrainingError(Exception):
    """
    Exception intended to be raised during training, if loss or parameter values in a tensor are torch.inf or torch.nan,
    to avoid wasting resources on a training routine with NaN cascade.
    """
    pass