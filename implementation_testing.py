###--- External Library Imports ---###
import torch

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from pathlib import Path


###--- Custom Imports ---###
from data_utils import DatasetBuilder, TimeSeriesDataset, AlignmentTS, custom_collate_fn, get_subset_by_label_status
from data_utils.data_filters import filter_by_machine

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

from models.transformer_ae.positional_encoding import PositionalEncoding

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    KMeansLoss,
)

from loss.clt_callbacks import LossTrajectoryObserver
from loss.topology_term import Topological
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from helper_tools import map_loader
from helper_tools.setup import create_normaliser

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
Test Functions - TensorDataset tests 
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def test_TensorDataset():
    dataset_kind = 'key'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]
    normaliser_kind = 'min_max'
    filter_condition = filter_by_machine('M_A')


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        #exclude_columns = exclude_columns,
        #filter_condition = filter_condition,
        #exclude_const_columns = False,
    )
    
    dataset = dataset_builder.build_dataset()
    #print(f'Dataset size: {len(dataset)}')
    #print(dataset.alignm.X_col_map)
    
    ###--- Get trivial 'model' results ---###
    labelled_subset = get_subset_by_label_status(dataset = dataset, labelled = True)
    indices = labelled_subset.indices
            
    y_data = dataset.y_data[indices, 1:]
    y_mean = y_data.mean(dim = 0, keepdim=True)

    y_deviation = (y_data - y_mean)
    l1_batch = torch.sum(torch.abs(y_deviation), dim = 1)
    l2_batch = torch.sqrt(torch.sum(y_deviation**2, dim = 1))

    l1_bar = torch.mean(l1_batch)
    l2_bar = torch.mean(l2_batch)

    mae_dim = torch.mean(torch.abs(y_deviation), dim = 0)
    mae = torch.mean(mae_dim)

    mse_dim = torch.mean(y_deviation**2, dim = 0)
    mse = torch.mean(mse_dim)

    print(
        f'L1_bar: {l1_bar}\n'
        f'L2_bar: {l2_bar}\n'
        f'MAE: {mae}\n'
        f'MSE: {mse}\n'
    )


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
Test Functions - Loss Term Composition
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def loss_term_composition():
    ae_base_weight = 0.5
    ete_regr_weight = 0.95


    loss_terms = {
        'L2': AEAdapter(LpNorm(p = 2)),
        'topo': Topological(p = 2),
        'kmeans': KMeansLoss(n_clusters = 5, latent_dim = 2),
        'Huber': RegrAdapter(Huber(delta = 1)),
    }

    ae_loss_base_name = 'L2'
    ae_loss_extra_name = 'topo'
    regr_loss_name = 'Huber'

    ae_clt = CompositeLossTerm(
        loss_terms = {ae_loss_base_name: loss_terms[ae_loss_base_name], ae_loss_extra_name: loss_terms[ae_loss_extra_name]}
    )

    ae_clt = WeightedCompositeLoss(
        composite_lt = ae_clt, 
        weights={ae_loss_base_name: ae_base_weight, ae_loss_extra_name: 1 - ae_base_weight}
    )

    ete_loss_terms = {
        'ae_loss': ae_clt,
        'regr_loss': loss_terms[regr_loss_name],
    }

    ete_clt = CompositeLossTerm(loss_terms = ete_loss_terms)
    ete_clt = WeightedCompositeLoss(
        composite_lt=ete_clt, 
        weights={'ae_loss': 1 - ete_regr_weight, regr_loss_name: ete_regr_weight}
    )

    ete_loss = Loss(ete_clt)
    ae_iso_loss = Loss(ae_clt)



"""
Test Functions - Transformer Approach Testing
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def positional_encoding_test():
    max_len = 1000
    d_model = 6

    pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    print(
        f'Positional encoding for max_len = {max_len}, d_model = {d_model}: \n'
        f'{pos_encoding.pe[:10]}\n'
    )



def transformer_approach():

    index_map = map_loader(Path('data/alignment_info/index_id_map.json'))
    alignment_ts = AlignmentTS(index_map = index_map)

    ts_dataset = TimeSeriesDataset(alignment = alignment_ts)

    # sample = ts_dataset[5]
    # print(
    #     f'Sample properties: \n'
    #     f'---------------------------------------------------------------\n'
    #     f'Type: \n{type(sample)}\n'
    #     f'Length: \n{len(sample)}\n'
    #     f'Shape: \n{sample.shape}\n'
    #     f'---------------------------------------------------------------\n'
    #     f'First 5 entries: \n{sample[:5]}\n'
    # )

    # Create DataLoader with collate function
    batch_size = 4
    loader = DataLoader(
        ts_dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True
    )

    for batch_idx, (padded_sequences, lengths) in enumerate(loader):
        print(f"Batch {batch_idx+1}")
        print(f"Padded sequences shape: {padded_sequences.shape}")
        print(f"Lengths tensor: {lengths}")
        print(f"Sample sequence (first in batch):\n{padded_sequences[0, :10]}")
        
        if batch_idx == 1:
            break



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


    ###--- TensorDataset ---###
    test_TensorDataset()


    ###--- Hyperparameter Opt. Cfgs ---###
    #test_hyperop_cfg()


    ###--- Transformer build Tests ---###
    #positional_encoding_test()
    #transformer_approach()
    