
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from itertools import product
from pathlib import Path
from collections import namedtuple

from info import process_metadata_cols, identifier_col



"""
DataSets - DataFrameNamedTupleDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""
DataPoint = namedtuple('DataPoint', ['metadata', 'X', 'y'])

class DataFrameNamedTupleDataset(Dataset):

    def __init__(self, X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):

        #merge X_data_df ID column with y_data_df to have indices alligned 
        y_data_df = X_data_df[[identifier_col]].merge(y_data_df, on=[identifier_col], how='left')

        self.metadata_df = X_data_df[process_metadata_cols]
        self.X_data_df = X_data_df.drop(columns=process_metadata_cols)

        self.X_column_names = self.X_data_df.columns.tolist() 
        self.y_column_names = y_data_df.columns.tolist() 

        self.y_data_df = y_data_df.drop(columns=[identifier_col])

        
    def __len__(self):

        return len(self.X_data_df)
    

    def __getitem__(self, ndx):
        
        metadata = self.metadata_df.iloc[ndx]
        X_data = self.X_data_df.iloc[ndx]
        y_data = self.y_data_df.iloc[ndx]

        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
        y_data = torch.tensor(y_data.to_numpy(), dtype=torch.float32)

        data = DataPoint(metadata.to_dict(), X_data, y_data)
        
        return data




"""
DataSets - DataFrameDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class DataFrameDataset(Dataset):

    def __init__(self, X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):

        self.X_column_names = X_data_df.columns.tolist() 
        self.y_column_names = y_data_df.columns.tolist() 

        #merge X_data_df ID column with y_data_df to have indices alligned 
        data_df = X_data_df.merge(y_data_df, on=[identifier_col], how='left')
        print(data_df.head(10))

        self.metadata_df_active = data_df[process_metadata_cols]
        self.X_data_df_active = data_df.drop(columns = process_metadata_cols + self.y_column_names)
        self.y_data_df_active = data_df[self.y_column_names]


    def __len__(self):

        pass
    

    def __getitem__(self, ndx):
        
        pass


"""
DataSets - TensorDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class TensorDataset(Dataset):

    def __init__(self, X_data_df: pd.DataFrame, y_data_df: pd.DataFrame):
        """
        Initializes the MetadataAwareDataset.

        Args:
            dataframe (pandas.DataFrame): The input dataframe containing both metadata and data.
            metadata_cols (list): List of column names representing the metadata.
        """

        y_data_df = X_data_df[[identifier_col]].merge(y_data_df, on=[identifier_col], how='left')
        y_data_df.drop(columns=[identifier_col], inplace=True)

        X_metadata_df = X_data_df[process_metadata_cols]
        X_data_df = X_data_df.drop(columns=process_metadata_cols)

        self.X_column_names = X_data_df.columns.tolist() 
        self.y_column_names = y_data_df.columns.tolist() 

        self.metadata = X_metadata_df.reset_index().set_index('index').to_dict(orient='index')
        self.X_data = torch.tensor(X_data_df.values, dtype=torch.float32)  
        self.y_data = torch.tensor(y_data_df.values, dtype=torch.float32)  


    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.X_data)

    def __getitem__(self, ndx):
        """
        Retrieves a data sample and its associated metadata.

        Args:
            ndx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - data (torch.Tensor): The tensor representation of the data sample.
                - metadata (dict): The metadata associated with the data sample.
                - column_names (list): The list of column names corresponding to the tensor's dimensions
        """
        return self.X_data[ndx], self.y_data[ndx], self.metadata[ndx], self.X_column_names, self.y_column_names









"""
Tests
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    

    pass