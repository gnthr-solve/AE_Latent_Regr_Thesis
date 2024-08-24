
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from itertools import product
from pathlib import Path

from datasets import DataFrameNamedTupleDataset, DataFrameDataset, TensorDataset

"""
Test Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def properties_test(X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):
    
    #--- Instantiate Dataset Subclass ---#
    dataset = DataFrameDataset(X_data_df, y_data_df)

    



def standard_test(X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):
    
    #--- Instantiate Dataset Subclass ---#
    dataset = TensorDataset(X_data_df, y_data_df)

    #--- Instantiate Dataset Subclass ---#
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    
    break_ndx = 0
    # Iterate over the DataLoader and print batches
    for batch_idx, batch_data in enumerate(data_loader):

        print(f"Batch Index: {batch_idx}")
        
        print(f"Type Batch Data: {type(batch_data)}")
        print(f"Batch Data: {batch_data}")

        if batch_idx == break_ndx:
            break



def tensor_tuple_test(X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):
    """
    names = ['metadata', 'X', 'y']
    """
    #--- Instantiate Dataset Subclass ---#
    dataset = DataFrameNamedTupleDataset(X_data_df, y_data_df)

    #--- Instantiate Dataset Subclass ---#
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    
    break_ndx = 0
    # Iterate over the DataLoader and print batches
    for batch_idx, batch_data in enumerate(data_loader):

        print(f"Batch Index: {batch_idx}")
        
        print(f"Type Batch Data: {type(batch_data)}")
        
        print(f"Batch Data Metadata: {batch_data.metadata}")
        print(f"Batch Data X: {batch_data.X}")
        print(f"Batch Data y: {batch_data.y}")


        if batch_idx == break_ndx:
            break




"""
Tests
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    #--- Load Pandas DataFrames ---#
    X_data_path = Path("data/X_data_plus_metadata.csv")
    #X_data_path = Path("data/X_data.csv")
    metadata_path = Path("data/metadata.csv")
    y_data_path = Path("data/y_data.csv")

    X_data_df = pd.read_csv(X_data_path)
    y_data_df = pd.read_csv(y_data_path)


    #--- Test Functions ---#
    properties_test(X_data_df, y_data_df)
    #standard_test(X_data_df, y_data_df)
    #tensor_tuple_test(X_data_df, y_data_df)