
import torch

from torch import Tensor

from .eval_visitor_abc import EvaluationVisitor

from ..evaluation import Evaluation
from ..model_output import ModelOutput


class OutputVisitor(EvaluationVisitor):
    """
    EvaluationVisitor sub-base-class to apply models to the data 
    and generate or modify ModelOutput.
    """
    def _get_data(self, eval: Evaluation) -> dict[str, Tensor]:
        """
        Retrieves the data assigned to the OutputVisitor by the EvalConfig from the Evaluation.

        Parameters
        ----------
            eval: Evaluation
                Evaluation instance accessed by the visitor.

        Returns:
            dict[str, Tensor]
                Dictionary of data tensors matching its data_key.
        """
        data_key = self.data_key
        return eval.test_data[data_key]



"""
Output Visitors - AEOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a deterministic or NVAE autoencoder model to the evaluation object.
"""
class AEOutputVisitor(OutputVisitor):

    def visit(self, eval: Evaluation):
        """
        Applies an autoencoder model to the test data 
        and inscribes a new ModelOutput instance into the evaluation.

        Intended for models with signature

            X_batch -> Z_batch, X_hat_batch

        AE models produce a latent space representation (Z_batch) and a reconstruction (X_hat_batch).
        """
        ae_model = eval.models['AE_model']

        data = self._get_data(eval)

        with torch.no_grad():

            Z_batch, X_hat_batch = ae_model(data['X_batch'])

            eval.model_outputs[self.output_name] = ModelOutput(
                Z_batch = Z_batch,
                X_hat_batch = X_hat_batch,
            )



"""
Output Visitors - VAEOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a variational autoencoder model to the evaluation object.
"""
class VAEOutputVisitor(OutputVisitor):

    def visit(self, eval: Evaluation):
        """
        Applies a VAE model to the test data 
        and inscribes a new ModelOutput instance into the evaluation.

        Intended for models with signature

            X_batch -> Z_batch, infrm_dist_params, genm_dist_params

        VAE models produce inference and generative model distribution parameters, 
        in addition to the latent representation, but no reconstruction.
        """
        ae_model = eval.models['AE_model']

        data = self._get_data(eval)

        with torch.no_grad():

            Z_batch, infrm_dist_params, genm_dist_params = ae_model(data['X_batch'])

            X_hat_batch, _ = genm_dist_params.unbind(dim = -1)

            eval.model_outputs[self.output_name] = ModelOutput(
                X_hat_batch = X_hat_batch,
                Z_batch = Z_batch,
                infrm_dist_params = infrm_dist_params,
                genm_dist_params = genm_dist_params,
            )



"""
Output Visitors - RegrOutputVisitor
-------------------------------------------------------------------------------------------------------------------------------------------
Inscribes the output of a regression model to the evaluation object.
"""
class RegrOutputVisitor(OutputVisitor):

    def visit(self, eval: Evaluation):
        """
        Applies regressor model to the test data 
        and inscribes either a new ModelOutput instance into the evaluation, or modifies a previous one.

        Depending on its 'mode' attribute this visitor 
        either applies the regressor to the input data (direct regression or 'iso' mode),
        or to the latent space representation ('composed' mode).
        """
        regressor = eval.models['regressor']
        
        data = self._get_data(eval = eval)

        if self.mode == 'composed':
            input_data = data['Z_batch']

        elif self.mode == 'iso':
            input_data = data['X_batch']

        with torch.no_grad():

            y_hat = regressor(input_data)

        if self.mode == 'composed':
            eval.model_outputs[self.output_name] = ModelOutput(**data, y_hat_batch = y_hat)
        
        elif self.mode == 'iso':
            eval.model_outputs[self.output_name] = ModelOutput(y_hat_batch = y_hat)

    
    def _get_data(self, eval: Evaluation) -> dict[str, Tensor]:
        
        if self.mode == 'composed':

            model_output = eval.model_outputs[self.output_name]
            data = model_output.to_dict()

        elif self.mode == 'iso':

            data = eval.test_data[self.data_key]
        
        return data