
import torch
import time
import pandas as pd

from abc import ABC, abstractmethod


"""
Observer ABC
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class IterObserver(ABC):

    @abstractmethod
    def __call__(epoch: int, iter_idx: int, **kwargs):
        pass


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


