
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


import matplotlib.pyplot as plt
from ..evaluation import Evaluation
from ..model_output import ModelOutput


class EvaluationVisitor(ABC):
    @abstractmethod
    def visit(self, evaluation: Evaluation):
        pass









