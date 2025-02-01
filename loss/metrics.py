import torch
import torch.linalg as tla

from torch import Tensor
from torch import nn

from .loss_term_classes import LossTerm



class RMSELoss(LossTerm):

    def __call__(self, y_batch: Tensor, y_hat_batch: Tensor, **tensors: Tensor) -> Tensor:
        
        return torch.sqrt(torch.mean((y_batch - y_hat_batch)**2, dim=1))




class R2Score(LossTerm):

    def __call__(self, y_batch: Tensor, y_hat_batch: Tensor, **tensors: Tensor) -> Tensor:

        errors = y_batch - y_hat_batch

        ss_res = torch.sum(errors**2, dim=1)
        ss_tot = torch.sum((y_batch - y_batch.mean(dim=0))**2, dim=1)

        return 1 - (ss_res / ss_tot)
