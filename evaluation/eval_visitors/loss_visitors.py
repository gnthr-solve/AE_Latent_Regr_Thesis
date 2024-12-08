
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

    def __init__(self, loss_term: LossTerm, eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_term = loss_term
        

    def visit(self, eval: Evaluation):
        
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

    def __init__(self, loss_term: LossTerm, eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_term = loss_term

    
    def visit(self, eval: Evaluation):

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

    def __init__(self, loss_term: LossTerm, eval_cfg: EvalConfig):
        super().__init__(eval_cfg = eval_cfg)

        self.loss_term = loss_term


    def visit(self, eval: Evaluation):

        eval_results = eval.results
        tensors = self._get_data(eval)

        with torch.no_grad():

            loss_batch = self.loss_term(**tensors)

            eval_results.losses[self.loss_name] = loss_batch
            eval_results.metrics[self.loss_name] = loss_batch.mean().item()


