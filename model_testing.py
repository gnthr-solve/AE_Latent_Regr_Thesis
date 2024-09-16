
import torch

from torch import Tensor
from torch.nn import Module

from models import (
    SimpleEncoder, 
    SimpleLinearReluEncoder,
    SimpleDecoder,
    SimpleLinearReluDecoder, 
    SimpleAutoencoder, 
    SimpleLoss,
    MeanLpLoss,
    RelativeMeanLpLoss,
)


"""
Test Functions - Module
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def module_properties_test():

    ###--- Models ---###
    latent_dim = 10

    encoder = SimpleLinearReluEncoder(latent_dim = latent_dim)
    decoder = SimpleLinearReluDecoder(latent_dim = latent_dim)

    model = SimpleAutoencoder(encoder = encoder, decoder = decoder)


    ###--- Properties of Composite Model ---###
    #NOTE: named_children, children iterates over the submodules directly defined in model, not nested.
    """
    module.py shows that module.named_children produces a dictionary-esque iterator,
    and module.children a list like iterator based on module.named_children
    """
    named_children = {name: child for name, child in model.named_children()}
    named_children_repr = ',\n'.join([f'{name}: \n{child}' for name, child in named_children.items()])
    print(
        f'model.named_children for model: \n{model}\n'
        f'-------------------------------------------------\n'
        f'{named_children_repr}\n'
        f'-------------------------------------------------\n\n'
    )


    #NOTE: .named_modules, .modules does nested iteration over all modules in the tree, including the model itself.
    """
    module.py shows that module.named_modules produces a dictionary-esque iterator,
    and module.modules a list like iterator analogous to module.named_children
    """
    named_modules = {name: module for name, module in model.named_modules()}
    named_modules_repr = ',\n'.join([f'{name}: \n{module}' for name, module in named_modules.items()])
    print(
        f'model.named_modules for model: \n{model}\n'
        f'-------------------------------------------------\n'
        f'{named_modules_repr}\n'
        f'-------------------------------------------------\n\n'
    )



"""
Test Functions Execution
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    #--- Module ---#
    module_properties_test()