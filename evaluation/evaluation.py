
import torch
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt

from data_utils import retrieve_metadata
from .model_output import ModelOutput

if TYPE_CHECKING:
    from .eval_visitors.eval_visitor_abc import EvaluationVisitor



@dataclass
class EvaluationResults:

    losses: dict[str, dict[str, torch.Tensor]] = field(default_factory = dict)
    metrics: dict[str, float] = field(default_factory = dict)

    plots: dict[str, plt.Figure] = field(default_factory = dict)



class Evaluation:
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
            
            self.test_data[kind] = kind_dict
            
            self.aligned_metadata[kind] = retrieve_metadata(mapping_idxs, self.dataset.metadata_df)
            
        self.test_data = test_data


    def accept(self, visitor: EvaluationVisitor):
        visitor.visit(self)
    

    def accept_sequence(self, visitors: list[EvaluationVisitor]):
        for visitor in visitors:
            self.accept(visitor)










