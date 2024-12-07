
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

    @abstractmethod
    def visit(self, evaluation: Evaluation):
        pass









