
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



"""
Setup - Helper function for potential instantiation using Hydra
-------------------------------------------------------------------------------------------------------------------------------------------
Taken from my research project, but not used here thus far.
"""
def retrieve_class(module_name, class_name):
    return importlib.import_module(module_name).__dict__[class_name]


"""
Setup - Retrieve Normalisers
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def create_normaliser(kind: str, epsilon: Optional[float] = None):
    """
    Short setup function to instantiate implemented normalisers based on name string.

    Parameters
    ----------
        kind: str
            Shorthand name of implemented normalisers.
        epsilon: Optional[float] = None
            Parameter necessary for MinMaxEpsNormaliser normaliser and irrelevant otherwise.

    Returns:
        normaliser | None
            Instantiated normaliser if string corresponds to implemented normaliser shorthand, 
            otherwise None - corresponds to using data without normalisation.
    """
    if kind == "min_max_eps":
        return MinMaxEpsNormaliser(epsilon=epsilon)
    
    elif kind == "min_max":
        return MinMaxNormaliser()
    
    elif kind == "z_score":
        return ZScoreNormaliser()
    
    else:
        return None
    


"""
Setup - Create Evaluation Metrics for Hyperoptimisation
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def create_eval_metric(name: str) -> LossTerm:
    """
    Short setup function to instantiate dedicated LossTerm's.
    Supported:
        - L2-norm and L1-norm based LossTerms
        - Huber LossTerm (currently only for regression)
        - Relative versions (divided by respective norm of input tensor)

    Parameters
    ----------
        name: str
            Shorthand name of a standard LossTerm.

    Returns:
        LossTerm
            Instantiated LossTerm instance.
            If the string ends with '_reconstr' the instance is wrapped into an AEAdapter,
            otherwise the instance is wrapped into a RegrAdapter.
    """
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