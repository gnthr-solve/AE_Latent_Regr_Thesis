###--- External Library Imports ---###
import torch

from torch import Tensor
from torch.nn import Module, LayerNorm
from torch.utils.data import DataLoader

from pathlib import Path
from typing import Iterable

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

from models.transformer_ae import PositionalEncoding, SelfAttentionHead, FFN

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
Test Functions - Helper Tools
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def test_nested_dict_str():
    from helper_tools import nested_dict_str
    d = {
        'A': 0,
        'B': {
            'B_10': (1,0),
            'B_11': {
                'B_20': (2,0),
                'B_21': (2,1),
                'B_22': (2,2),
            }
        },
        'C': {
            'C_10': (1,0),
            'B_11': (1,1),
        },
    }

    print(nested_dict_str(d))




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



def loss_term_conf():
    from dataclasses import dataclass
    from loss import LossTerm, CompositeLossTerm, WeightedCompositeLoss

    ###--- (C)LT Config DCs ---###
    @dataclass
    class LTConfig:
        name: str
        type_name: str
        type_kwargs: dict


    @dataclass
    class AdaptationConfig:
        name: str
        term_names: list[str]


    ###--- (C)LT Builder ---###
    class LTBuilder:

        def create_from_config(self, cfgs: list[LTConfig]):
            
            loss_terms = {
                cfg.name: LossTerm._registry.get(cfg.type_name)(**cfg.type_kwargs)
                for cfg in cfgs
            }
            return loss_terms


        def create_from_clt_configs(self, clt_cfgs):
            clts = {}

            for clt_cfg in clt_cfgs:
                member_terms = self.create_from_config(clt_cfg.terms)

                clts[clt_cfg.name] = CompositeLossTerm(member_terms)
            
            return clts


    ###--- create_from_config LT Test ---###
    cfgs = [
        LTConfig(name = 'L2-Error', type_name = 'LpNorm', type_kwargs={'p': 2}),
        LTConfig(name = 'L1-Error', type_name = 'LpNorm', type_kwargs={'p': 1}),
        LTConfig(name = 'Topo', type_name = 'Topological', type_kwargs={'p': 2}),
        LTConfig(name = 'KMeans', type_name = 'KMeansLoss', type_kwargs={'n_clusters': 5, 'latent_dim': 2}),
    ]

    lt_builder = LTBuilder()
    loss_terms = lt_builder.create_from_config(cfgs = cfgs)

    for name, lt in loss_terms.items():
        print(
            f'LT {name}: \n'
            f'----------------------\n'
            f'{lt}\n'
            f'{lt.__dict__}\n'
            f'----------------------\n'
        )


    ###--- create_from_config CLT Test ---###
    composition_cfgs = {
        'AE_terms':[
            LTConfig(name = 'L2-Error', type_name = 'LpNorm', type_kwargs={'p': 2}),
            LTConfig(name = 'KMeans', type_name = 'KMeansLoss', type_kwargs={'n_clusters': 5, 'latent_dim': 2}),
        ],
        'VAE_terms':[
            LTConfig(name = 'LL', type_name = 'GaussianDiagLL', type_kwargs={}),
            LTConfig(name = 'KLD', type_name = 'GaussianAnaKLDiv', type_kwargs={}),
        ]
    }

    composition_loss_terms = {
        name: lt_builder.create_from_config(cfgs = cfgs)
        for name, cfgs in composition_cfgs.items()
    }

    clts = {name: CompositeLossTerm(terms) for name, terms in composition_loss_terms.items()}
    
    for name, lt in clts.items():
        print(
            f'LT {name}: \n'
            f'----------------------\n'
            f'{lt}\n'
            f'{lt.__dict__}\n'
            f'----------------------\n'
        )



def loss_term_conf_dict_builder():
    from dataclasses import dataclass
    from loss import LossTerm, CompositeLossTerm, WeightedCompositeLoss

    from helper_tools import flatten_nested_dict_keys, nested_dict_str
    
    ###--- (C)LT Config DCs ---###
    @dataclass
    class LTConfig:
        name: str
        type_name: str
        type_kwargs: dict


    @dataclass
    class GroupingConfig:
        name: str
        term_names: list[str]


    ###--- (C)LT Builder ---###
    class LTDictBuilder:

        def __init__(self):
            self.loss_terms = {}

        def create_from_config(self, lt_cfgs: list[LTConfig]):
            
            loss_terms = {
                cfg.name: LossTerm._registry.get(cfg.type_name)(**cfg.type_kwargs)
                for cfg in lt_cfgs
            }
            
            self.loss_terms.update(loss_terms)


        def generate_grouping(self, grouping_cfg: GroupingConfig):

            lt_group = {}

            for lt_name in grouping_cfg.term_names:

                lt_group[lt_name] = self.loss_terms.pop(lt_name)
            
            self.loss_terms[grouping_cfg.name] = lt_group


        def generate_groupings(self, grouping_cfgs: Iterable[GroupingConfig]):

            for grouping_cfg in grouping_cfgs:
                
                self.generate_grouping(grouping_cfg = grouping_cfg)


        def construct_CLT(self, loss_terms: dict = {}, members: list[str] = None):

            loss_terms = self.loss_terms if not loss_terms else loss_terms
            
            members = flatten_nested_dict_keys(loss_terms) if members is None else members

            clt_loss_terms = {}
            for name, element in loss_terms.items():
                if isinstance(element, dict):
                    clt_loss_terms[name] = self.construct_CLT(loss_terms = element, members = members)
                elif name in members:
                    clt_loss_terms[name] = element
                else:
                    continue
                

            return CompositeLossTerm(clt_loss_terms)
        

        def construct_CLT_alt1(self, loss_terms: dict = {}, members: list[str] = None):

            loss_terms = self.loss_terms if not loss_terms else loss_terms
            
            clt_loss_terms = {}
            for name, element in loss_terms.items():
                if isinstance(element, dict):
                    clt_loss_terms[name] = self.construct_CLT(loss_terms = element, members = members)
                elif members is not None:
                    if name in members:
                        clt_loss_terms[name] = element
                    else:
                        continue
                else:
                    clt_loss_terms[name] = element

            return CompositeLossTerm(clt_loss_terms)


    ###--- LTDictBuilder Test ---###
    cfgs = [
        LTConfig(name = 'L2-Error', type_name = 'LpNorm', type_kwargs={'p': 2}),
        LTConfig(name = 'L1-Error', type_name = 'LpNorm', type_kwargs={'p': 1}),
        LTConfig(name = 'Topo', type_name = 'Topological', type_kwargs={'p': 2}),
        LTConfig(name = 'KMeans', type_name = 'KMeansLoss', type_kwargs={'n_clusters': 5, 'latent_dim': 2}),
        LTConfig(name = 'LL', type_name = 'GaussianDiagLL', type_kwargs={}),
        LTConfig(name = 'KLD', type_name = 'GaussianAnaKLDiv', type_kwargs={}),
    ]

    grouping_cfgs = [
        GroupingConfig(name = 'VAE Loss', term_names = ['LL', 'KLD']),
        GroupingConfig(name = 'Regr. Loss', term_names = ['L2-Error', 'L1-Error']),
    ]

    lt_builder = LTDictBuilder()
    lt_builder.create_from_config(lt_cfgs = cfgs)
    # print(
    #     f'Builder loss_terms after create_from_config: \n'
    #     f'----------------------\n'
    #     f'{lt_builder.loss_terms}\n'
    #     f'----------------------\n'
    # )

    for i, group_cfg in enumerate(grouping_cfgs, start = 1):

        lt_builder.generate_grouping(grouping_cfg = group_cfg)

        # print(
        #     f'Builder loss_terms after group {i}: \n'
        #     f'----------------------\n'
        #     f'{lt_builder.loss_terms}\n'
        #     f'----------------------\n'
        # )

    print(nested_dict_str(lt_builder.loss_terms))

    members = ['L2-Error', 'L1-Error', 'LL', 'KLD']
    main_clt = lt_builder.construct_CLT(members = members)
    #main_clt = lt_builder.construct_CLT(members = None)
    
    print(
        f'Main CLT by construct_CLT method: \n'
        f'----------------------\n'
        f'{main_clt}\n'
        f'{main_clt.__dict__}\n'
        f'----------------------\n'
    )


    
    


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

    max_len = 1000
    d_model = 108
    d_inner = 50
    d_k = 20

    pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    attention_head = SelfAttentionHead(d_model = d_model, d_k = d_k)
    ffn = FFN(d_model = d_model, d_inner = d_inner)
    attention_layer_norm = LayerNorm(d_model)
    ffn_layer_norm = LayerNorm(d_model)

    for batch_idx, (X_seq_batch, lengths) in enumerate(loader):

        print(
            f'Batch {batch_idx+1}: \n'
            f'------------------------------------------------------------\n'
            f'Padded sequences shape: {X_seq_batch.shape}\n'
            f'Lengths tensor:\n {lengths}\n'
            f'Sample sequence (first in batch):\n{X_seq_batch[0, :5, :5]}\n'
            f'------------------------------------------------------------\n\n'
        )
        
        ###--- Positional Encoding ---###
        X_seq_batch = pos_encoding(X_seq_batch)

        print(
            f'Batch {batch_idx+1} after PE: \n'
            f'------------------------------------------------------------\n'
            f'Encoded sequences shape: {X_seq_batch.shape}\n'
            f'Sample sequence (first in batch):\n{X_seq_batch[0, :5, :5]}\n'
            f'------------------------------------------------------------\n\n'
        )

        ###--- Attention Head Residual Connection ---###
        X_seq_batch = attention_layer_norm(X_seq_batch + attention_head(input = X_seq_batch, lengths = lengths))

        print(
            f'Batch {batch_idx+1} after Layer Norm(Residual Connection Attention Head):\n'
            f'------------------------------------------------------------\n'
            f'X_seq_batch shape: {X_seq_batch.shape}\n'
            f'Sample sequence (first in batch):\n{X_seq_batch[0, :5, :5]}\n'
            f'------------------------------------------------------------\n\n'
        )

        ###--- FFN pass ---###
        X_seq_batch = ffn_layer_norm(X_seq_batch + ffn(X_seq_batch))

        print(
            f'Batch {batch_idx+1} after LayerNorm(Residual Connection FFN):\n'
            f'------------------------------------------------------------\n'
            f'X_seq_batch shape: {X_seq_batch.shape}\n'
            f'Sample sequence (first in batch):\n{X_seq_batch[0, :5, :5]}\n'
            f'------------------------------------------------------------\n\n'
        )

        if batch_idx == 0:
            break



"""
Test Functions - Execution
-------------------------------------------------------------------------------------------------------------------------------------------
"""

if __name__=="__main__":

    ###--- Helper Tools ---###
    #test_nested_dict_str()


    ###--- Module ---###
    #module_properties_test()


    ###--- Product Regressor ---###
    #product_regr_test()


    ###--- DNN Funnel layout ---###
    #test_DNN_layout()


    ###--- TensorDataset ---###
    #test_TensorDataset()


    ###--- Hyperparameter Opt. Cfgs ---###
    #test_hyperop_cfg()


    ###--- Loss Terms ---###
    #loss_term_conf()
    loss_term_conf_dict_builder()


    ###--- Transformer build Tests ---###
    #positional_encoding_test()
    #transformer_approach()
    