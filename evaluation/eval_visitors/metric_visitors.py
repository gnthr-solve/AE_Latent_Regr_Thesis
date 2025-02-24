
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .eval_visitor_abc import EvaluationVisitor

from ..evaluation import Evaluation
from ..eval_config import EvalConfig
from ..model_output import ModelOutput

from loss import LossTerm


"""
Metric Visitors
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossStatisticsVisitor(EvaluationVisitor):
    """
    Analyzes distribution of loss values
    """
    def __init__(self, loss_name: str, eval_cfg: EvalConfig, quantiles:list[int] = []):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_name = loss_name
        self.quantiles = quantiles or [25, 75, 95]


    def visit(self, eval: Evaluation):
        loss_batch = eval.results.losses[self.loss_name]
        
        stats = {
            'std': torch.std(loss_batch),
            'median': torch.median(loss_batch),
            'MAD': torch.median(torch.abs(loss_batch - torch.median(loss_batch))),
            'min': torch.min(loss_batch),
            'max': torch.max(loss_batch),
            **{f'p{q}': torch.quantile(loss_batch, q/100) for q in self.quantiles}
        }

        eval.results.metrics.update({
            f'{self.loss_name}_{k}': v.item() for k,v in stats.items()
        })
