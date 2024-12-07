
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .eval_visitor_abc import EvaluationVisitor

from ..evaluation import Evaluation
from ..model_output import ModelOutput

from loss import LossTerm

"""
Loss Visitors - ReconstrLossVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ReconstrLossVisitor(EvaluationVisitor):

    def __init__(self, loss_term: LossTerm, name: str):
        self.loss_term = loss_term
        self.name = name


    def visit(self, eval: Evaluation):

        with torch.no_grad():

            for kind, data in eval.test_data.items():

                X_batch = data['X_batch']

                model_output = eval.model_outputs[f'ae_{kind}']

                X_hat_batch = model_output.X_hat_batch

                loss_batch = self.loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch)




"""
Loss Visitors - RegrLossVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RegrLossVisitor(EvaluationVisitor):

    def __init__(self, loss_term: LossTerm, name: str):
        self.loss_term = loss_term
        self.name = name

    
    def visit(self, eval: Evaluation):

        y_batch = eval.test_data['labelled']['y_batch']

        model_output = eval.model_outputs[f'regression']

        y_hat_batch = model_output.y_hat_batch
        
        with torch.no_grad():

            loss_batch = self.loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch)




"""
Loss Visitors - ComposedLossVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ComposedLossVisitor(EvaluationVisitor):

    def __init__(self, loss_term: LossTerm, name: str):
        self.loss_term = loss_term
        self.name = name


    def visit(self, eval: Evaluation):

        y_batch = eval.test_data['labelled']['y_batch']

        model_output = eval.model_outputs[f'composed']

        y_hat_batch = model_output.y_hat_batch

        with torch.no_grad():

            loss_batch = self.loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch)




class LossTermVisitor(EvaluationVisitor):

    def __init__(self, loss_term: LossTerm, name: str):
        self.loss_term = loss_term
        self.name = name


    def visit(self, eval: Evaluation):

        with torch.no_grad():

            for kind, data in eval.test_data.items():

                model_output = eval.model_outputs[self.name]

                tensors = {**data, **model_output.to_dict()}

                loss_batch = self.loss_term(**tensors)
