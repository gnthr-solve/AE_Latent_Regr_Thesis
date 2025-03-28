
import torch
import time
import pandas as pd

from torch import Tensor

from abc import ABC, abstractmethod
from collections import defaultdict

"""
Simple Training Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class TrainingObserver:
    
    def __init__(self, store_batches: bool = False, device: str = 'cpu'):
        self.store_batches = store_batches
        self.device = device
        self.history = defaultdict(list)
        

    def __call__(self, name: str, loss_batch: Tensor):
        
        detached = loss_batch.detach().to(self.device)

        if self.store_batches:
            self.history[name].append(detached)

        else:
            self.history[name].append(detached.mean().item())
            

