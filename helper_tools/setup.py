
import torch
import torch.nn as nn

from torch.nn import Module
from torch import Tensor

from models import VAE, AE, VarEncoder, VarDecoder, LinearEncoder, LinearDecoder
from abc import ABC, abstractmethod

from .torch_general import weights_init

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
