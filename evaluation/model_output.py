
from torch import Tensor

from dataclasses import dataclass
from typing import Any, Optional

from dataclasses import asdict


@dataclass
class ModelOutput:
    """
    ModelOutput Container class.
    For connected models Visitors write to the same output container.

    Example:
        Autoencoder: X_batch -> Z_batch, X_hat_batch
        Regressor: Z_batch -> y_hat_batch
    """
    Z_batch: Optional[Tensor] = None
    X_hat_batch: Optional[Tensor] = None
    
    infrm_dist_params: Optional[Tensor] = None
    genm_dist_params: Optional[Tensor] = None

    y_hat_batch: Optional[Tensor] = None
    
    def to_dict(self) -> dict[str, Tensor]:
        """
        Returns only non-None entries
        """
        return {k: v for k, v in asdict(self).items() if v is not None}
    



