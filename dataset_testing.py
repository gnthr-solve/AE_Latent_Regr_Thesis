
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from itertools import product
from pathlib import Path

from data_utils import DataFrameDataset, TensorDataset, SplitSubsetFactory

"""
Test Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def DataFrameDataset_test(joint_data_df: pd.DataFrame):
    
    #--- Instantiate Dataset Subclass ---#
    batch_size = 200

    dataset = DataFrameDataset(joint_data_df)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    
    break_idx = 0
    # Iterate over the DataLoader and print batches
    for b_idx, (X_batch, y_batch) in enumerate(dataloader):

        #print(f"Batch Index: {b_idx}")
        
        print(f"X batch: {X_batch}")
        print(f"X batch shape: {X_batch.shape}")
        print(f"y Batch: {y_batch}")
        print(f"y batch shape: {y_batch.shape}")

        if b_idx == break_idx:
            break




def TensorDataset_test(X_data: Tensor, y_data: Tensor, metadata_df: pd.DataFrame):
    
    #--- Instantiate Dataset Subclass ---#
    dataset = TensorDataset(X_data, y_data, metadata_df)


    #--- Verify Index Alignment ---#
    test_indices = [i for i in range(20)]
    y_data = dataset.y_data
    X_data = dataset.X_data
    metadata_df = dataset.metadata_df
    print(
        f'Data index values at test indices:\n'
        f'=================================================\n'
        f'y_data:\n{y_data[test_indices, 0]}\n'
        f'-------------------------------------------------\n'
        f'X_data:\n{X_data[test_indices, 0]}\n'
        f'-------------------------------------------------\n'
        f'metadata_df:\n{metadata_df.loc[test_indices]}\n'
        f'=================================================\n'
    )


    #--- Instantiate DataLoader ---#
    # batch_size = 200
    # dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    # break_idx = 0
    # # Iterate over the DataLoader and print batches
    # for b_idx, (X_batch, y_batch) in enumerate(dataloader):

    #     #print(f"Batch Index: {b_idx}")
        
    #     print(f"X batch: {X_batch}")
    #     print(f"X batch shape: {X_batch.shape}")
    #     print(f"y Batch: {y_batch}")
    #     print(f"y batch shape: {y_batch.shape}")

    #     if b_idx == break_idx:
    #         break




def SubsetFactory_test(X_data: Tensor, y_data: Tensor, metadata_df: pd.DataFrame):
    
    #--- Instantiate Dataset Subclass ---#

    dataset = TensorDataset(X_data, y_data, metadata_df)
    
    split_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)

    #splits = split_factory.create_splits_alpha()
    splits = split_factory.create_splits()
    
    size = 0
    for key, split in splits.items():

        split_indices = split.indices
        split_y_data = dataset.y_data[split_indices]

        print(
            f'For split-kind {key}:\n'
            f'=================================================\n'
            f'Type split: \n{type(split)}\n'
            f'Size split: \n{len(split)}\n'
            f'y_data has nan: \n{split_y_data.isnan().any(dim = 1).any()}\n'
            f'y_data all nan: \n{split_y_data.isnan().any(dim = 1).all()}\n'
            f'=================================================\n\n'
        )

        size += len(split)


    print(
        f'Dataset size: {len(dataset)}\n'
        f'Sum of split sizes: {size}\n'
    )




def subset_test(X_data: Tensor, y_data: Tensor, metadata_df: pd.DataFrame):
    
    #--- Instantiate Dataset & Subset's ---#
    dataset = TensorDataset(X_data, y_data, metadata_df)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    X_test = test_dataset.dataset.X_data[test_dataset.indices] 




"""
Tests
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    data_dir = Path("./data")
    tensor_dir = data_dir / "tensors"

    ###--- Load Pandas DataFrames & Tensors ---###
    joint_data_df = pd.read_csv(data_dir / "data_joint.csv")
    metadata_df = pd.read_csv(data_dir / "metadata.csv")

    X_data: torch.Tensor = torch.load(f = tensor_dir / 'X_data_tensor.pt')
    y_data: torch.Tensor = torch.load(f = tensor_dir / 'y_data_tensor.pt')

    #--- Test Functions ---#
    #DataFrameDataset_test(joint_data_df)
    #TensorDataset_test(X_data, y_data, metadata_df)
    SubsetFactory_test(X_data, y_data, metadata_df)