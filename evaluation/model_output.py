
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


from dataclasses import asdict

@dataclass
class ModelOutput:

    Z_batch: Optional[torch.Tensor] = None
    X_hat_batch: Optional[torch.Tensor] = None
    
    infrm_dist_params: Optional[torch.Tensor] = None
    genm_dist_params: Optional[torch.Tensor] = None

    y_hat_batch: Optional[torch.Tensor] = None
    
    def to_dict(self) -> dict[str, torch.Tensor]:
        """Returns only non-None entries"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    



@dataclass
class ModelOutputPrime:
    """Unified container for model outputs"""
    
    latent: Optional[torch.Tensor] = None
    reconstruction: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None
    
    infrm_dist_params: Optional[torch.Tensor] = None
    genm_dist_params: Optional[torch.Tensor] = None
    

    @classmethod
    def from_ae(cls, Z: torch.Tensor, X_hat: torch.Tensor) -> 'ModelOutputPrime':
        return cls(latent=Z, reconstruction=X_hat)
    
    @classmethod
    def from_vae(cls, Z: torch.Tensor, infrm_dist_params: torch.Tensor, 
                 genm_dist_params: torch.Tensor) -> 'ModelOutputPrime':
        return cls(
            latent=Z,
            infrm_dist_params = infrm_dist_params,
            genm_dist_params = genm_dist_params,
            reconstruction= genm_dist_params[..., 0]
        )
    
    @classmethod
    def from_regressor(cls, y_hat: torch.Tensor) -> 'ModelOutputPrime':
        return cls(prediction=y_hat)

   