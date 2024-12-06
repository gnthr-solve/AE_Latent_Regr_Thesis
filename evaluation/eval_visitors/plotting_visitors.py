
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt

from .eval_visitor_abc import EvaluationVisitor
from ..evaluation import Evaluation
from ..model_output import ModelOutput


class LatentVisualizationVisitor(EvaluationVisitor):
    def visit(self, eval: Evaluation):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        eval.figures['latent'] = fig
        
        ae_output = eval.model_outputs['ae_labelled']
        indices = eval.test_data['labelled']['indices']
        metadata = eval.metadata_df.iloc[indices]
        
        self._plot_reconstruction_error(axes[0], ae_output, eval.metrics['reconstr_loss'])
        self._plot_temporal_distribution(axes[1], ae_output, metadata['timestamp'])