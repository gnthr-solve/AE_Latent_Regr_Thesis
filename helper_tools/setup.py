
import torch
import torch.nn as nn
import importlib

from abc import ABC, abstractmethod
from typing import Any, Optional
from torch.nn import Module
from torch import Tensor

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser
from loss import *

from .torch_general import weights_init

def retrieve_class(module_name, class_name):
    return importlib.import_module(module_name).__dict__[class_name]


"""
Retrieve Normalisers
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def create_normaliser(kind: str, epsilon: Optional[float] = None):

    if kind == "min_max_eps":
        return MinMaxEpsNormaliser(epsilon=epsilon)
    
    elif kind == "min_max":
        return MinMaxNormaliser()
    
    elif kind == "z_score":
        return ZScoreNormaliser()
    
    else:
        return None
    


"""
Create Evaluation Metrics for Hyperoptimisation
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def create_eval_metric(name: str) -> LossTerm:

    ###--- Regr Eval Metrics ---###
    if name == 'L2-norm':
        return RegrAdapter(LpNorm(p = 2))
     
    if name == 'Rel_L2-norm':
        return RegrAdapter(RelativeLpNorm(p = 2))
    
    if name == 'L1-norm':
        return RegrAdapter(LpNorm(p = 1))
    
    if name == 'Rel_L1-norm':
        return RegrAdapter(RelativeLpNorm(p = 1))
    
    if name == 'Huber':
        return RegrAdapter(Huber())
    
    if name == 'Rel_Huber':
        return RegrAdapter(RelativeHuber())
    
    
    ###--- Reconstr Eval Metrics ---###
    if name == 'L2-norm_reconstr':
        return AEAdapter(LpNorm(p = 2))
     
    if name == 'Rel_L2-norm_reconstr':
        return AEAdapter(RelativeLpNorm(p = 2))
    
    if name == 'L1-norm_reconstr':
        return AEAdapter(LpNorm(p = 1))
    
    if name == 'Rel_L1-norm_reconstr':
        return AEAdapter(RelativeLpNorm(p = 1))