
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
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.history = defaultdict(list)
        

    def __call__(self, name: str, tensor_batch: Tensor):
        
        detached = tensor_batch.detach().to(self.device)

        self.history[name].append(detached)

            

