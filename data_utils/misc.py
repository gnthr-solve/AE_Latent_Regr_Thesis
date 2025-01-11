
import torch
import pandas as pd
import numpy as np
import json

from torch import Tensor
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

from itertools import product
from pathlib import Path
from collections import namedtuple
from dataclasses import dataclass, field

from .datasets import TensorDataset
from .alignment import Alignment

from preprocessing.normalisers import Normaliser

from .dataset_builder import DatasetBuilder
from .split_factory import SplitSubsetFactory

"""
Data Helper Tools - Setup build Test
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def setup_dataset_build(
        kind: str, normaliser: Normaliser| None, exclude_columns: list[str], 
        train_size: float,
    ) -> tuple[TensorDataset, SplitSubsetFactory]:

    dataset_builder = DatasetBuilder(
        kind = kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns
    )
    
    dataset = dataset_builder.build_dataset()
   
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = train_size)

    return dataset, subset_factory



"""
Data Helper Tools - Get full subset based on label status
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def get_subset_by_label_status(dataset: TensorDataset, labelled: bool = True) -> Subset:
    
    y_data = dataset.y_data
    y_isnan = y_data[:, 1:].isnan().all(dim = 1)

    if labelled:
        indices = torch.where(~y_isnan)[0].tolist()
    else:
        indices = torch.where(y_isnan)[0].tolist()
    
    return Subset(dataset=dataset, indices=indices)




"""
Data Helper Tools - Load Tensor segments
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def load_tensor_segment(tensor_path: Path, indices: list[int]) -> Tensor:

    tensor = torch.load(tensor_path, weights_only = True)
    tensor_segment = tensor[indices]

    return tensor_segment




"""
Data Helper Tools - Build Dataframe from Dataset Segment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def build_dataset_segment_dataframe(dataset: TensorDataset, indices: list[int], from_mapping_idxs: bool = False):

    if not from_mapping_idxs:
        #indices is a list from e.g. a Subset instance
        mapping_idxs = dataset.X_data[indices, 0].tolist()
    else:
        mapping_idxs = indices

    matched_metadata_df = dataset.metadata_df.query('mapping_idx.isin(@mapping_idxs)')

    # Convert indices to a tensor
    mapping_indices = torch.tensor(mapping_idxs, dtype=dataset.X_data.dtype)

    # Unsqueezing allows for broadcasting the comparison operation which then results in a matrix mask tensor, reduced by any
    data_mask = (dataset.X_data[:, 0].unsqueeze(-1) == mapping_indices).any(dim=-1)

    X_data = dataset.X_data[data_mask, 1:]
    y_data = dataset.y_data[data_mask, 1:]

    X_col_names = [dataset.alignm.X_col_map[i] for i in range(1, X_data.shape[1] + 1)]
    y_col_names = [dataset.alignm.y_col_map[i] for i in range(1, y_data.shape[1] + 1)]

    X_data_df = pd.DataFrame(X_data, index=matched_metadata_df.index, columns = X_col_names)
    y_data_df = pd.DataFrame(y_data, index=matched_metadata_df.index, columns = y_col_names)

    reconstructed_df = matched_metadata_df.join([X_data_df, y_data_df])
    
    return reconstructed_df



"""
Data Helper Tools - tensor to df
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def X_tensor_to_df(X_data: Tensor, alignment, metadata: pd.DataFrame) -> pd.DataFrame:

    index_map = alignment.index_map
    col_map = alignment.X_col_map

    indices = X_data[:, 0].tolist()
    df_dict = {'WAFER_ID': [index_map[int(idx)] for idx in indices]}
    df_dict.update({col_map[i]: X_data[:, i].tolist() for i in range(1, X_data.shape[1])})

    df = pd.DataFrame(df_dict)
    df = df.merge(metadata, on = 'WAFER_ID')

    return df
    



"""
Data Helper Tools - retrieve metadata for indices
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def retrieve_metadata(indices, metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a Sequence of indices by a Subset of the Dataset, retrieve the corresponding metadata.
    """
    contained_ids = [int(idx) for idx in indices]
    mask = metadata_df['mapping_idx'].isin(contained_ids)

    return metadata_df[mask]



"""
Data Helper Tools - Storage
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def remove_columns_and_update_mapping(tensor: Tensor, mapping: dict[int, str], keys_to_remove: list[str]):

    keys_to_keep = [key for key in mapping.values() if key not in keys_to_remove]
    indices_to_keep = [int(k) for k, v in mapping.items() if v in keys_to_keep]

    new_tensor = tensor[:, indices_to_keep]
    new_mapping = {i: v for i, v in enumerate(keys_to_keep)}

    return new_tensor, new_mapping




"""
Data Helper Tools - Combine labelled and unlabelled subsets for isolated AE training
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def combine_subsets(subset_l: Subset, subset_ul: Subset) -> Subset:
    
    combined_indices = torch.cat([
        torch.tensor(subset_l.indices),
        torch.tensor(subset_ul.indices)
    ])

    return Subset(subset_l.dataset, combined_indices.tolist())




"""
Data Helper Tools - Custom Collate for TimeSeriesDataset
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def custom_collate_fn(batch: list[Tensor]):
    """
    Returns padded sequences and original lengths for timeseries for batching
    """
    lengths = [ts.shape[0] for ts in batch]

    padded_sequences = pad_sequence(batch, batch_first=True)
    
    return padded_sequences, torch.tensor(lengths)