###--- External Library Imports ---###
import torch

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from pathlib import Path


###--- Custom Imports ---###
from data_utils import TimeSeriesDataset, AlignmentTS, custom_collate_fn

from models.encoders import (
    LinearEncoder,
)

from models.decoders import (
    LinearDecoder,
)
from models.var_encoders import VarEncoder
from models.var_decoders import VarDecoder
from models.layer_blocks import LinearFunnel, ExponentialFunnel
from models.regressors import LinearRegr, ProductRegr, FunnelDNNRegr
from models import AE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    Loss,
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from helper_tools import map_loader


"""
Test Functions - Module
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def module_properties_test():

    ###--- Models ---###
    latent_dim = 10

    encoder = LinearEncoder(latent_dim = latent_dim)
    decoder = LinearDecoder(latent_dim = latent_dim)

    model = AE(encoder = encoder, decoder = decoder)


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
Test Functions - Product Regressor
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def product_regr_test():

    ###--- Tensors ---###
    latent_dim = 5
    batch_size = 3
    # z = torch.randint(1, 10, (latent_dim,))
    # Z_batch = torch.randint(1, 10, (batch_size, latent_dim))
    z = torch.rand((latent_dim,))
    #Z_batch = torch.rand((batch_size, latent_dim))
    Z_batch = torch.randn((batch_size, latent_dim))

    print(
        f'Input Tensors: \n'
        f'-------------------------------------------------\n'
        # f'z: \n{z}\n'
        # f'z shape: {z.shape}\n'
        # f'-------------------------------------------------\n'
        f'Z_batch: \n{Z_batch}\n'
        f'Z_batch shape: {Z_batch.shape}\n'
        f'-------------------------------------------------\n'
    )

    # print(
    #     f'Input Tensors unsqueezed: \n'
    #     f'-------------------------------------------------\n'
    #     f'z: \n{z.unsqueeze(dim = -1)}\n'
    #     f'z shape: {z.unsqueeze(dim = -1).shape}\n'
    #     f'-------------------------------------------------\n'
    #     f'Z_batch: \n{Z_batch.unsqueeze(dim = -1)}\n'
    #     f'Z_batch shape: {Z_batch.unsqueeze(dim = -1).shape}\n'
    #     f'-------------------------------------------------\n'
    # )


    ###--- Model ---###
    y_dim = 2
    product_regr = ProductRegr(latent_dim = latent_dim, y_dim = y_dim)

    ###--- Forward Pass ---###
    #y_hat = product_regr(z)
    Y_hat_batch = product_regr(Z_batch)
    print(
        f'Output Tensors: \n'
        f'-------------------------------------------------\n'
        #f'y_hat: \n{y_hat}\n'
        #f'y_hat shape: {y_hat.shape}\n'
        #f'-------------------------------------------------\n'
        f'Y_hat_batch: \n{Y_hat_batch}\n'
        f'Y_hat_batch shape: {Y_hat_batch.shape}\n'
    )




"""
Test Functions - DNN
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def test_DNN_layout():
    #regressor = FunnelDNNRegr(input_dim = 200, n_layers = 3)
    linear_funnel = LinearFunnel(input_dim = 200, output_dim=2, n_layers = 5)
    exp_funnel = ExponentialFunnel(input_dim = 200, output_dim=2)




"""
Test Functions - Hyperparam configs
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def test_hyperop_cfg():

    from hyperoptim.experiment_cfgs import (
        linear_regr_iso_cfg, 
        deep_NN_regr_cfg,
        shallow_NN_regr_cfg, 
        vae_iso_cfg, 
        ae_linear_joint_epoch_cfg,
        ae_deep_joint_epoch_cfg,
    )

    print(linear_regr_iso_cfg)




"""
Test Functions - Transformer Approach Testing
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def transformer_approach():

    index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
    alignment_ts = AlignmentTS(index_map = index_map)

    ts_dataset = TimeSeriesDataset(alignment = alignment_ts)

    sample = ts_dataset[5]
    print(
        f'Sample properties: \n'
        f'---------------------------------------------------------------\n'
        f'Type: \n{type(sample)}\n'
        f'Length: \n{len(sample)}\n'
        f'Shape: \n{sample.shape}\n'
        f'---------------------------------------------------------------\n'
        f'First 10 entries: \n{sample[:10]}\n'
    )



"""
Test Functions - Execution
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    ###--- Module ---###
    #module_properties_test()


    ###--- Product Regressor ---###
    #product_regr_test()


    ###--- DNN Funnel layout ---###
    #test_DNN_layout()


    ###--- Hyperparameter Opt. Cfgs ---###
    #test_hyperop_cfg()


    ###--- Transformer build Tests ---###
    transformer_approach()
    