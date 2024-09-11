
import torch
import numpy as np
import pandas as pd
import re
import importlib

from itertools import product
from functools import wraps



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


def print_dict(d: dict):
    
    dict_strs = [
        f"{key}: {value}"
        for key, value in d.items()
    ]

    print('\n'.join(dict_strs))




def print_iter_types(iterable_obj):
    
    iter_types = [
        type(item)
        for item in iterable_obj
    ]

    print(
        f"Type of iterable object: {type(iterable_obj)}\n"
        f"Types in iterable object:\n"
        f"----------------------------------------------\n"
        f"{iter_types}\n"
        f"----------------------------------------------\n"
    )




"""
String Discriminator
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class StringDiscriminator:
    """
    Callable meant to search a string any substring in list of interest and to be applied as a boolean check in comprehensions.

    Args:
        list_of_interest: List of strings to search for.
        filter_out: Boolean flag to indicate whether to return a positive or a negative result for a match.
    """

    def __init__(self, list_of_interest: list[str], filter_out: bool = False):

        self.list_of_interest = list_of_interest
        self.filter_out = filter_out
    

    def __call__(self, string: str) -> bool:

        result = any(re.search(ref_string, string) for ref_string in self.list_of_interest)

        return self.filter_out ^ result
    



