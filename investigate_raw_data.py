
import torch
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from itertools import product
from pathlib import Path

from helper_tools import print_dict, StringDiscriminator

"""
Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def count_NaN_cols_X(df_X: pd.DataFrame):
    
    num_NaN = lambda col: df_X[col].isna().sum()

    col_nan_counts = {
        col: num_NaN(col)
        for col in df_X.columns
        if num_NaN(col) > 0
    }
    
    print_dict(col_nan_counts)



def NaN_rows_X(df_X: pd.DataFrame):

    df_isna = df_X.isna().any(axis = 1)

    nan_rows = df_X[df_isna].index

    print(nan_rows)



def identify_const_cols_X(df_X: pd.DataFrame):
    
    const_cols = {
        col: df_X[col].unique()
        for col in df_X.columns
        if df_X[col].nunique() == 1
    }
    
    const_cols_strs = [
        f"{col}: {values}"
        for col, values in const_cols.items()
    ]

    const_cols_str = '\n'.join(const_cols_strs)

    print(
        f'Number of Constant Columns: {len(const_cols)} \n'
        'Constant Columns: \n'
        '-----------------\n'
        f'{const_cols_str}\n'
        '-----------------\n'
    )



def identify_const_zero_cols_X(df_X: pd.DataFrame):
    
    const_cols = {
        col: float(df_X[col].unique())
        for col in df_X.columns
        if df_X[col].nunique(dropna = False) == 1
    }
    
    const_zero_cols = [
        col
        for col, value in const_cols.items()
        if value == 0
    ]

    discr = StringDiscriminator(list_of_interest = ['SetValue'])
    const_zero_set_value_cols = [col for col in const_zero_cols if discr(col)]
    const_zero_value_cols = [col for col in const_zero_cols if not discr(col)]


    const_zero_set_value_str = '\n'.join(const_zero_set_value_cols)
    const_zero_value_str = '\n'.join(const_zero_value_cols)

    print(
        f'Number of Constant Zero Columns: {len(const_cols)} \n'
        f'Columns with constant zero setting ({len(const_zero_set_value_cols)}): \n'
        '-----------------\n'
        f'{const_zero_set_value_str}\n'
        '-----------------\n'
        f'Columns with constant zero value ({len(const_zero_value_cols)}): \n'
        '-----------------\n'
        f'{const_zero_value_str}\n'
        '-----------------\n'
    )



def set_value_cols(df_X: pd.DataFrame):

    discr = StringDiscriminator(list_of_interest = ['SetValue'])

    sv_cols = [col for col in df_X.columns if discr(col)]
    #print(sv_cols)

    df_sv = df_X[sv_cols]

    df_sv_values = {
        col: df_sv[col].unique()
        for col in sv_cols
        if df_sv[col].unique().size < 5
    }

    print_dict(df_sv_values)



def completion_time_hist(df_X: pd.DataFrame):
    
    compl_times = df_X['Time_ptp']

    short_bound = 110
    long_bound = 140
    
    short_mask = (compl_times < short_bound)  
    long_mask = (compl_times > long_bound)

    short_compl_times = compl_times[short_mask]
    long_compl_times = compl_times[long_mask]
    ordinary_compl_times = compl_times[~(short_mask | long_mask)]
    
    fig, axes = plt.subplots(
        nrows = 1, 
        ncols = 3,
    )

    fig.suptitle('Completion Time Histograms')
    fig.set_size_inches(15, 5)

    axes[0].hist(short_compl_times, bins=50)
    axes[1].hist(ordinary_compl_times, bins=50)
    axes[2].hist(long_compl_times, bins=50)
    
    axes[0].set_title(f'Short (<{short_bound}), n={len(short_compl_times)}')
    axes[1].set_title(f'Ordinary (between {short_bound} and {long_bound}), n={len(ordinary_compl_times)}')
    axes[2].set_title(f'Long (>{long_bound}), n={len(long_compl_times)}')

    plt.show()



if __name__=="__main__":

    #--- DataFrame Paths ---#
    joint_data_path = Path("data/data_joint.csv")
    metadata_path = Path("data/metadata.csv")
    X_data_path = Path("data/X_data.csv")
    y_data_path = Path("data/y_data.csv")

    X_data_df = pd.read_csv(X_data_path)
    y_data_df = pd.read_csv(y_data_path)
    metadata_df = pd.read_csv(metadata_path)

    #--- Investigate DataFrames ---#
    #count_NaN_cols_X(X_data_df)
    #NaN_rows_X(X_data_df)
    #identify_const_cols_X(X_data_df)
    #identify_const_zero_cols_X(X_data_df)
    #set_value_cols(X_data_df)
    #completion_time_hist(X_data_df)


    #--- Investigate MinMax NaN Cols ---#
    with open('info_files/min_max_nan_cols.yaml', 'r') as file:
        cols = yaml.safe_load(file)

    print(X_data_df[cols].drop_duplicates())