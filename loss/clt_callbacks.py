
import torch

import logging

from torch import Tensor

from collections import defaultdict
from typing import Callable, Optional

from helper_tools import AbortTrainingError, no_grad_decorator
from .loss_classes import LossTerm, CompositeLossTerm

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
Callbacks - Loss Trajectory Tracker
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class LossTrajectoryTracker:

    def __init__(self):
        self.history = defaultdict(list)
        

    @no_grad_decorator
    def __call__(self, name: str, loss_batch: Tensor):

        mean_loss = loss_batch.mean().item()
        self.history[name].append(mean_loss)
            
