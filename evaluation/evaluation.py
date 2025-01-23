
import torch
import pandas as pd

from torch import Tensor
from torch.utils.data import Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt

from data_utils import retrieve_metadata
from .model_output import ModelOutput

if TYPE_CHECKING:
    from .eval_visitors.eval_visitor_abc import EvaluationVisitor


"""
Evaluation - Evaluation Results Container
-------------------------------------------------------------------------------------------------------------------------------------------
"""
@dataclass
class EvaluationResults:
    """
    Container class to store results, 
    that are produced by applying EvaluationVisitor's to an Evaluation instance.
    """
    losses: dict[str, Tensor] = field(default_factory = lambda: dict())
    metrics: dict[str, float] = field(default_factory = lambda: dict())

    plots: dict[str, plt.Figure] = field(default_factory = lambda: dict())




"""
Evaluation - Primary Evaluation Class
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Evaluation:
    """
    Main Evaluation class. Follows the Visitor design pattern.
    Initialises incorporating the TensorDataset instance, the Subset's of the dataset, 
    and the models that are to be evaluated.
    Evaluation prepares and stores the data, the outputs of the models applied to the data
    and the results of the evaluation.
    Accepts EvaluationVisitors that operate on the data stored within it.

    Input Parameters
    ----------
        dataset: TensorDataset
            Instance of the dataset that was used for training.
        subsets: dict[str, Subset]
            Dictionary with names as keys and Subset instances as values.
            E.g. key 'labelled' with Subset of all labelled samples.
        models: dict[str, torch.nn.Module]
            Dictionary of (named) trained models that are to be evaluated.
    """
    def __init__(
            self, 
            dataset: TensorDataset, 
            subsets: dict[str, Subset],
            models: dict[str, torch.nn.Module],
        ):

        self.dataset = dataset
        self.models = models
        self.metadata_df = dataset.metadata_df

        self.aligned_metadata: dict[str, pd.DataFrame] = {}
        self._prepare_data(subsets)
        
        # Containers for results
        self.model_outputs: dict[str, ModelOutput] = {}
        self.results: EvaluationResults = EvaluationResults()
        
    
    def _prepare_data(self, subsets: dict[str, Subset]) -> dict[str, Tensor]:
        """
        Prepares the data for evaluation.
        Retrieves the tensors for input data 'X_batch' and the labels 'y_batch', 
        that match the Subset.indices from the dataset, and stores them separately in a dictionary.
        Furthermore it aligns the extracted mapping indices with the metadata 
        and stores the aligned metadata in a separate dictionary.
        Does not have return values but modifies the Evaluation instance inplace.

        Parameters
        ----------
            subsets: dict[str, Subset]
                Dictionary with names as keys and Subset instances as values.
        
        Output
        ----------
            test_data: dict[str, dict[str, Tensor]]
                Dictionary of dictionaries where the keys correspond to the Subset names
                and the values are dictionaries of separated tensors matching the Subset.indices.
            aligned_metadata: dict[str, pd.DataFrame]
                Dictionary with Subset names as keys and aligned metadata DataFrames as values.
                Metadata DataFrames represent the metadata of the samples matching Subset.indices. 
        """
        test_data = {}

        for kind, subset in subsets.items():

            kind_dict = {}
            indices = subset.indices
            
            # Get data
            X_data = self.dataset.X_data[indices]
            y_data = self.dataset.y_data[indices]
            
            # Store mapping indices and actual data separately
            mapping_idxs = X_data[:, 0].tolist()
            kind_dict['mapping_indices'] = mapping_idxs
            kind_dict['X_batch'] = X_data[:, 1:]
            kind_dict['y_batch'] = y_data[:, 1:]
            
            test_data[kind] = kind_dict
            
            self.aligned_metadata[kind] = retrieve_metadata(mapping_idxs, self.dataset.metadata_df)
            
        self.test_data = test_data


    def accept(self, visitor: 'EvaluationVisitor'):
        visitor.visit(self)
    

    def accept_sequence(self, visitors: list['EvaluationVisitor']):
        for visitor in visitors:
            self.accept(visitor)










