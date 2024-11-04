
import torch
import numpy as np

from torch import Tensor
from torch.nn import Module

from itertools import product
from functools import wraps

import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from helper_tools import plot_training_losses, plot_param_norms

"""
Autoencoder Parameter Observer
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class AEParameterObserver:

    def __init__(self):
        
        self.losses = []
        self.encoder_param_values: dict[str, list] = {}
        self.encoder_param_grads: dict[str, list] = {}

        self.decoder_param_values: dict[str, list] = {}
        self.decoder_param_grads: dict[str, list] = {}
    

    def __call__(self, loss: Tensor, ae_model: Module):

        self.losses.append(loss.item())

        for name, param in ae_model.named_parameters():
            
            if name.startswith('encoder'):
                param_values_list = self.encoder_param_values.get(name, [])
                param_grads_list = self.encoder_param_grads.get(name, [])
            
            else:
                param_values_list = self.decoder_param_values.get(name, [])
                param_grads_list = self.decoder_param_grads.get(name, [])
            
            # param_values_list.append(param.data.norm().item())
            # param_grads_list.append(param.grad.norm().item())
            param_values_list.append(param.data.norm().tolist())
            param_grads_list.append(param.grad.norm().tolist())

            if name.startswith('encoder'):
                self.encoder_param_values[name] = param_values_list
                self.encoder_param_grads[name] = param_grads_list

            else:
                self.decoder_param_values[name] = param_values_list
                self.decoder_param_grads[name] = param_grads_list
            
            if torch.isnan(param.data).any():
                print(f"{name} contains NaN values")
                raise StopIteration
            
            if torch.isinf(param.data).any():
                print(f"{name} contains Inf values")
                raise StopIteration
    

    def plot_results(self):

        title: str = "Training Characteristics",

        mosaic_layout = [
            ['losses', 'losses'],
            ['encoder_values', 'encoder_grads'],
            ['decoder_values', 'decoder_grads'],
        ]

        fig = plt.figure(figsize=(14, 7), layout = 'constrained')  
        axs = fig.subplot_mosaic(mosaic_layout)

        axs['losses'] = plot_training_losses(losses = self.losses, axes = axs['losses'])

        axs['encoder_values'] = plot_param_norms(norms = self.encoder_param_values, axes = axs['encoder_values'])
        axs['encoder_grads'] = plot_param_norms(norms = self.encoder_param_grads, axes = axs['encoder_grads'])

        axs['decoder_values'] = plot_param_norms(norms = self.decoder_param_values, axes = axs['decoder_values'])
        axs['decoder_grads'] = plot_param_norms(norms = self.decoder_param_grads, axes = axs['decoder_grads'])

        fig.suptitle(title)

        plt.show()

