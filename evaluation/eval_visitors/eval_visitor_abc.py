
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


from ..evaluation import Evaluation
from ..eval_config import EvalConfig
from ..model_output import ModelOutput


class EvaluationVisitor(ABC):

    def __init__(self, eval_cfg: EvalConfig):
        self.eval_cfg = eval_cfg

    @property
    def data_key(self) -> str:
        return self.eval_cfg.data_key
    
    @property
    def output_name(self) -> str:
        return self.eval_cfg.output_name
    
    @property
    def mode(self) -> str:
        return self.eval_cfg.mode
    
    @property
    def loss_name(self) -> str:
        return self.eval_cfg.loss_name
    
    @property
    def description(self) -> str:
        return self.eval_cfg.description
    
    @abstractmethod
    def visit(self, evaluation: Evaluation):
        pass









