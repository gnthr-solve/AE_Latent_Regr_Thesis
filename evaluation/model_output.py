
import torch

from torch import Tensor

from dataclasses import dataclass
from typing import Any, Optional

from dataclasses import asdict

@dataclass
class ModelOutput:

    Z_batch: Optional[Tensor] = None
    X_hat_batch: Optional[Tensor] = None
    
    infrm_dist_params: Optional[Tensor] = None
    genm_dist_params: Optional[Tensor] = None

    y_hat_batch: Optional[Tensor] = None
    
    def to_dict(self) -> dict[str, Tensor]:
        """Returns only non-None entries"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    



