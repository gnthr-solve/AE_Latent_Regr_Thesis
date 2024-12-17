
import torch
import torch.nn as nn
import importlib


from torch.nn import Module
from torch import Tensor

from models import VAE, AE, VarEncoder, VarDecoder, LinearEncoder, LinearDecoder
from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser
from abc import ABC, abstractmethod
from typing import Any, Optional

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
Model Factory Experiment
-------------------------------------------------------------------------------------------------------------------------------------------

"""
class ModelFactory(ABC):

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def setup_loss(self):
        pass




class AEBaseModelFactory(ModelFactory):

    def __init__(self, 
            model_type: VAE|AE, 
            input_dim: int, 
            latent_dim: int, 
            n_layers_e: int, 
            n_layers_d: int,
            n_dist_params: int = None,
            activation: str = 'ReLU'
        ):
        
        self.model_type = model_type

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_dist_params = n_dist_params

        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d
        self.activation = activation


    def setup_encoder(self):

        if self.model_type == VAE:

            encoder = VarEncoder(
                input_dim = self.input_dim,
                latent_dim = self.latent_dim,
                n_dist_params = self.n_dist_params,
                n_layers = self.n_layers_e,
                activation = self.activation
            )
        
        elif self.model_type == AE:

            encoder = LinearEncoder(
                input_dim = self.input_dim,
                latent_dim = self.latent_dim,
                n_layers = self.n_layers_e,
                activation = self.activation
            )

        weights_init(encoder, self.activation)

        return encoder


    def setup_decoder(self):

        if self.model_type == VAE:

            decoder = VarDecoder(
                output_dim = self.input_dim,
                latent_dim = self.latent_dim,
                n_dist_params = self.n_dist_params,
                n_layers = self.n_layers_d,
                activation = self.activation
            )
        
        elif self.model_type == AE:

            decoder = LinearDecoder(
                output_dim = self.input_dim,
                latent_dim = self.latent_dim,
                n_layers = self.n_layers_d,
                activation = self.activation
            )

        weights_init(decoder, self.activation)

        return decoder


    def setup_model(self):

        encoder = self.setup_encoder()
        decoder = self.setup_decoder()
        ae_model = self.model_type(encoder, decoder)

        return ae_model
