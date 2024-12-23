
import torch
import pandas as pd
import numpy as np
import yaml
import json

from itertools import product
from pathlib import Path


def drop_unnamed_y():
    """
    Reads:
        mean_MR.csv (corresponding to y data)

    Writes:
        mean_MR.csv (corresponding to y data)
    """

    ###--- Paths and Settings ---###
    data_dir = Path("./data")
    
    file_name = 'mean_MR.csv'
    
    mean_MR_df = pd.read_csv(data_dir /file_name)

    print(mean_MR_df.columns)

    mean_MR_df = mean_MR_df.drop(columns= ['Unnamed: 0.1', 'Unnamed: 0'])

    print(mean_MR_df.columns)

    mean_MR_df.to_csv(data_dir/file_name, index = False)



def drop_unnamed_Xmax():
    """
    Reads:
        apc_dataset_max.csv (corresponding to Xmax data)

    Writes:
        apc_dataset_max.csv (corresponding to Xmax data)
    """

    ###--- Paths and Settings ---###
    data_dir = Path("./data/raw")
    
    file_name = 'apc_dataset_max.csv'
    
    df = pd.read_csv(data_dir /file_name, low_memory = False)

    print(df.columns)

    df = df.drop(columns= ['Unnamed: 0'])

    print(df.columns)

    df.to_csv(data_dir/file_name, index = False)