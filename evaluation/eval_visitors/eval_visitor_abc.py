
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
    """
    EvaluationVisitor abstract base class following the Visitor pattern.
    An Evaluation instance accepts EvaluationVisitor's that access its attributes 
    and modify the assigned containers in place.

    Base class initialises with an EvalConfig and relays attribute requests to it.
    """
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
    def description(self) -> str:
        return self.eval_cfg.description
    
    @abstractmethod
    def visit(self, eval: Evaluation):
        """
        Abstract method to operate on an Evaluation instance and modify it in place.
        EvaluationVisitors access the Evaluation attributes and produce model outputs, metrics and plots.
        
        Parameters
        ----------
            eval: Evaluation
                Evaluation instance accessed by the visitor.
        """
        pass









