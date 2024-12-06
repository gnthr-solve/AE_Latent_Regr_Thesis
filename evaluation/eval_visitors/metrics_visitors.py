
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


class MetricsVisitor(EvaluationVisitor):
    def visit(self, eval: Evaluation):
        for kind, output in eval.model_outputs.items():
            if output.X_hat_batch is not None:
                eval.metrics[f'{kind}_reconstr_loss'] = self._compute_reconstr_loss(
                    output.X_hat_batch,
                    eval.test_data[kind]['X_data']
                )