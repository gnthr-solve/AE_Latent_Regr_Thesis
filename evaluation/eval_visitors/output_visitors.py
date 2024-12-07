
import torch

from torch import Tensor

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .eval_visitor_abc import EvaluationVisitor

from ..evaluation import Evaluation
from ..model_output import ModelOutput


"""
Output Visitors - AEOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a deterministic or NVAE autoencoder model to the evaluation object.
"""
class AEOutputVisitor(EvaluationVisitor):

    def visit(self, eval: Evaluation):

        ae_model = eval.models['AE_model']

        with torch.no_grad():

            for kind, data in eval.test_data.items():

                Z_batch, X_hat_batch = ae_model(data['X_batch'])

                eval.model_outputs[f'ae_{kind}'] = ModelOutput(
                    Z_batch = Z_batch,
                    X_hat_batch = X_hat_batch,
                )



"""
Output Visitors - VAEOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a variational autoencoder model to the evaluation object.
"""
class VAEOutputVisitor(EvaluationVisitor):

    def visit(self, eval: Evaluation):

        ae_model = eval.models['AE_model']

        with torch.no_grad():

            for kind, data in eval.test_data.items():

                Z_batch, infrm_dist_params, genm_dist_params = ae_model(data['X_batch'])

                eval.model_outputs[f'ae_{kind}'] = ModelOutput(
                    Z_batch = Z_batch,
                    infrm_dist_params = infrm_dist_params,
                    genm_dist_params = genm_dist_params,
                )



"""
Output Visitors - RegrOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a regression model to the evaluation object.
"""
class RegrOutputVisitor(EvaluationVisitor):

    def visit(self, eval: Evaluation, mode: str = 'composed'):

        regressor = eval.models['regressor']
        
        if mode == 'composed':
            ae_output = eval.model_outputs['ae_labelled']
            input_data = ae_output.Z_batch

        else:
            input_data = eval.test_data['labelled']['X_batch']

        with torch.no_grad():

            y_hat = regressor(input_data)

        eval.model_outputs['regression'] = ModelOutput(y_hat_batch = y_hat)