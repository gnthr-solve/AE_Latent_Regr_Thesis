
import torch
import time
import pandas as pd

from abc import ABC, abstractmethod


"""
Observer ABC
-------------------------------------------------------------------------------------------------------------------------------------------
Abstract Base Class for Observers that are integrated into a TrainingProcedure and called at every training-iteration within the procedure.
"""
class IterObserver(ABC):

    @abstractmethod
    def __call__(epoch: int, iter_idx: int, **kwargs):
        pass


"""
NOTE: There are two kinds of IterObserver.
    1. IterObservers that track batch-independent tensors.
    2. IterObservers that track batch-dependent tensors.

They require different approaches for initialisation and assignment.

For 1.
If the observer is independent of the size of the batch, then both initialisation and assignment can be done directly,
using the number of epochs and the number of iterations in the init, and the epoch and iteration index in slicing-assignment.

For 2.
If the observer is dependent on the size of the batch then the initialisation should be done with the size of the dataset,
and the batch size needs to be saved for determining the slice indices in the assignment.
Otherwise the last assignment will be incomplete if the dataset size is not a multiple of the batch size.
"""




"""
Subject Role Interface
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Subject:

    def __init__(self):
        self.observers = []


    def register_observers(self, *observers: IterObserver):
        self.observers.extend(observers)


    def notify_observers(self, epoch: int, iter_idx: int, **kwargs):

        for observer in self.observers:
        
            observer(epoch, iter_idx, **kwargs)


