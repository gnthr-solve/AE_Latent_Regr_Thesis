
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


class LossVisitor(EvaluationVisitor):
    """
    EvaluationVisitor sub-base-class to apply LossTerms and metrics to ModelOutput instances.
    """
    def _get_data(self, eval: Evaluation) -> dict[str, Tensor]:

        data_key = self.data_key
        data = eval.test_data[data_key]

        model_output = eval.model_outputs[self.output_name]

        return {**data, **model_output.to_dict()}
    

"""
Loss Visitors - ReconstrLossVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class ReconstrLossVisitor(LossVisitor):

    def __init__(self, loss_term: LossTerm, loss_name: str, eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_term = loss_term
        self.loss_name = loss_name
        

    def visit(self, eval: Evaluation):
        """
        Calculates reconstruction losses for AE model outputs via a LossTerm instance.

        Produces both the complete loss batch, inscribed in losses, and the mean loss, inscribed in metrics.
        """
        eval_results = eval.results

        data = self._get_data(eval)

        with torch.no_grad():

            X_batch = data['X_batch']
            X_hat_batch = data['X_hat_batch']

            loss_batch = self.loss_term(X_batch = X_batch, X_hat_batch = X_hat_batch)

            eval_results.losses[self.loss_name] = loss_batch
            eval_results.metrics[self.loss_name] = loss_batch.mean().item()





"""
Loss Visitors - RegrLossVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class RegrLossVisitor(LossVisitor):

    def __init__(self, loss_term: LossTerm, loss_name: str, eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_term = loss_term
        self.loss_name = loss_name
    
    def visit(self, eval: Evaluation):
        """
        Calculates losses for regression model outputs via a LossTerm instance.

        Produces both the complete loss batch, inscribed in losses, and the mean loss, inscribed in metrics.
        """
        eval_results = eval.results

        data = self._get_data(eval)
        y_batch = data['y_batch']
        y_hat_batch = data['y_hat_batch']
        
        with torch.no_grad():

            loss_batch = self.loss_term(y_batch = y_batch, y_hat_batch = y_hat_batch)

            eval_results.losses[self.loss_name] = loss_batch
            eval_results.metrics[self.loss_name] = loss_batch.mean().item()




"""
Loss Visitors - Generalisation Attempt
-------------------------------------------------------------------------------------------------------------------------------------------
"""

class LossTermVisitor(LossVisitor):

    def __init__(self, loss_terms: dict[str, LossTerm], eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_terms = loss_terms


    def visit(self, eval: Evaluation):
        """
        Applies a dictionary of named LossTerms to the output of a model or model composition.

        Produces both the complete loss batches, inscribed in losses, and the mean losses, inscribed in metrics.
        """
        eval_results = eval.results
        tensors = self._get_data(eval)

        with torch.no_grad():
            
            for name, loss_term in self.loss_terms.items():

                loss_batch = loss_term(**tensors)

                eval_results.losses[name] = loss_batch
                eval_results.metrics[name] = loss_batch.mean().item()


