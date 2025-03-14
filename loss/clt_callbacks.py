
import torch

import logging

from torch import Tensor

from collections import defaultdict
from typing import Callable, Optional

from helper_tools import AbortTrainingError, no_grad_decorator

logger = logging.getLogger(__name__)



"""
Composite LossTerm Callbacks - Loss Spike Detector
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossSpikeDetektor:
    """
    Callback for CompositeLossTerm
    Logs a warning if drastic changes in the loss values, based on exceeding a threshold, 
    occur for a particular LossTerm. Intended especially for VAE, where spikes were frequently observed.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.last_value = None

    
    @no_grad_decorator
    def __call__(self, name: str, loss_batch: Tensor):

        current = loss_batch.detach().mean().item()
        
        if self.last_value is not None:

            if abs(current - self.last_value) > self.threshold:
                logger.warning(
                    f"Sudden change in {name}: {self.last_value:.4f} -> {current:.4f}"
                )
        
        self.last_value = current




"""
Composite LossTerm Callbacks - Numerical Stability Monitor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class NumericalStabilityMonitor:
    """
    Callback for CompositeLossTerm
    Raises an error when any values of an assigned loss term become NaN or Inf.
    """
    
    @no_grad_decorator
    def __call__(self, name: str, loss_batch: Tensor):

        if torch.isnan(loss_batch).any():
            msg = f"NaN detected in loss term '{name}'"
            logger.error(msg)
            
            raise AbortTrainingError(msg)
                
        if torch.isinf(loss_batch).any():
            msg = f"Inf detected in loss term '{name}'"
            logger.error(msg)
            
            raise AbortTrainingError(msg)




"""
Callbacks - Loss Trajectory Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossTrajectoryObserver:
    """
    Callback version of list-based Observer pattern.
    """
    def __init__(self, store_batches: bool = False, device: str = 'cpu'):
        self.store_batches = store_batches
        self.device = device
        self.history = defaultdict(list)
        

    @no_grad_decorator
    def __call__(self, name: str, loss_batch: Tensor):
        """
        Callback method. Detaches loss_batch and stores it in a list for each name.
        """
        detached = loss_batch.detach().to(self.device)

        if self.store_batches:
            self.history[name].append(detached)

        else:
            self.history[name].append(detached.mean().item())
            

    def get_history(self, concat: bool = False) -> dict[str, list[Tensor] | Tensor]:
        """
        Return stored loss history. 
        If 'concat == True' return a dict of concatenated tensors.
        Otherwise a dict of lists of tensors.
        If 'store_batches' was set to True concatenation might raise an error,
        as the last batch might mismatch, leading to a tensor with uneven shape.
        """
        if concat:
            history = {
                name: torch.cat(batches) 
                if self.store_batches else torch.tensor(batches, device=self.device, dtype=torch.float32)
                for name, batches in self.history.items()
            }

        else:
            history = dict(self.history)

        return history