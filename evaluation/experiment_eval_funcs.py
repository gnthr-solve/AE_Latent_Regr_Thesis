
import torch

from pathlib import Path

from data_utils import TensorDataset, DatasetBuilder, get_subset_by_label_status

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, DNNRegr
from models import AE, VAE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from helper_tools.setup import create_eval_metric, create_normaliser


from .evaluation import Evaluation
from .eval_config import EvalConfig
from .eval_visitors import AEOutputVisitor, RegrOutputVisitor, LossTermVisitor




"""
Experiment Evaluation Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def evaluation_linear_regr(model_dir: Path, data_kind: str, normaliser_kind: str, **kwargs) -> Evaluation:

    model_paths = {model_path.stem: model_path for model_path in list(model_dir.glob("*.pt"))} 
    print(model_paths)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")


    ###--- Model Setup ---###
    regressor = LinearRegr(latent_dim = input_dim)

    regressor.load_state_dict(torch.load(model_paths['regressor']))
    regressor.eval()

    # print(
    #     f'Regressor:\n'
    #     f'-------------------------------------\n'
    #     f'{regressor}\n'
    #     f'-------------------------------------\n'
    # )


    ###--- Evaluation ---###
    labelled_subset = get_subset_by_label_status(dataset = dataset, labelled = True)

    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'labelled': labelled_subset},
        models = {'regressor': regressor},
    )

    eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
    }

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso')

    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        LossTermVisitor(loss_terms = eval_metrics, eval_cfg = eval_cfg)
    ]

    evaluation.accept_sequence(visitors = visitors)

    return evaluation
    


def evaluation_dnn_regr(model_dir: Path, data_kind: str, normaliser_kind: str, best_model_params: dict, **kwargs) -> Evaluation:

    model_paths = {model_path.stem: model_path for model_path in list(model_dir.glob("*.pt"))} 
    print(model_paths)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")
    

    regressor = DNNRegr(
        input_dim = input_dim,
        output_dim = 2,
        **best_model_params,
    )

    regressor.load_state_dict(torch.load(model_paths['regressor']))
    regressor.eval()

    print(
        f'Regressor:\n'
        f'-------------------------------------\n'
        f'{regressor}\n'
        f'-------------------------------------\n'
    )


    ###--- Evaluation ---###
    labelled_subset = get_subset_by_label_status(dataset = dataset, labelled = True)

    evaluation = Evaluation(
        dataset = dataset,
        subsets = {'labelled': labelled_subset},
        models = {'regressor': regressor},
    )

    eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
    }

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso')

    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        LossTermVisitor(loss_terms = eval_metrics, eval_cfg = eval_cfg)
    ]

    evaluation.accept_sequence(visitors = visitors)
    
    return evaluation
    



def evaluation_ae_linear(model_dir: Path, data_kind: str, normaliser_kind: str, best_model_params: dict, **kwargs) -> Evaluation:

    ###--- Model Paths ---###

    model_paths = {model_path.stem: model_path for model_path in list(model_dir.glob("*.pt"))} 
    print(model_paths)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")
    

    ###--- Best Parameters ---###
    latent_dim = best_model_params['latent_dim']
    
    n_layers = best_model_params['n_layers']
    activation = best_model_params['activation']


    ###--- Instantiate Model ---###
    encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    
    ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = LinearRegr(latent_dim = latent_dim)

    ae_model.load_state_dict(torch.load(model_paths['ae_model']))
    regressor.load_state_dict(torch.load(model_paths['regressor']))
    
    ae_model.eval()
    regressor.eval()

    ###--- Print Model State ---###
    # param_dict = {
    #     child_name: {
    #         name: param 
    #         for name, param in child.named_parameters()}
    #     for child_name, child in ae_model.named_children()
    # }
    # print(param_dict)
    print(
        f'AE Model:\n'
        f'-------------------------------------\n'
        f'{ae_model}\n'
        f'-------------------------------------\n'
        f'Regressor:\n'
        f'-------------------------------------\n'
        f'{regressor}\n'
        f'-------------------------------------\n'
    )


    ###--- Evaluation ---###
    subsets = {
        'unlabelled': get_subset_by_label_status(dataset = dataset, labelled = False),
        'labelled': get_subset_by_label_status(dataset = dataset, labelled = True),
    }

    evaluation = Evaluation(
        dataset = dataset,
        subsets = subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    regr_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
    }

    ae_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm_reconstr', 'Rel_L2-norm_reconstr', 'L1-norm_reconstr', 'Rel_L1-norm_reconstr']
    }

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)

    return evaluation




def evaluation_ae_deep(model_dir: Path, data_kind: str, normaliser_kind: str, best_model_params: dict, **kwargs) -> Evaluation:

    model_paths = {model_path.stem: model_path for model_path in list(model_dir.glob("*.pt"))} 
    print(model_paths)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")
    

    ###--- Best Parameters ---###
    latent_dim = best_model_params['latent_dim']
    
    n_layers = best_model_params['n_layers']
    n_fixed_layers = best_model_params['n_fixed_layers']
    fixed_layer_size = best_model_params['fixed_layer_size']
    n_funnel_layers = best_model_params['n_funnel_layers']
    activation = best_model_params['activation']


    ###--- Instantiate Model ---###
    encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = n_layers, activation = activation)
    
    ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = DNNRegr(
        input_dim = latent_dim,
        n_fixed_layers = n_fixed_layers,
        fixed_layer_size = fixed_layer_size,
        n_funnel_layers = n_funnel_layers,
        activation = activation,
    )

    ae_model.load_state_dict(torch.load(model_paths['ae_model']))
    regressor.load_state_dict(torch.load(model_paths['regressor']))
    
    ae_model.eval()
    regressor.eval()
    print(
        f'AE Model:\n'
        f'-------------------------------------\n'
        f'{ae_model}\n'
        f'-------------------------------------\n'
        f'Regressor:\n'
        f'-------------------------------------\n'
        f'{regressor}\n'
        f'-------------------------------------\n'
    )


    ###--- Evaluation ---###
    subsets = {
        'unlabelled': get_subset_by_label_status(dataset = dataset, labelled = False),
        'labelled': get_subset_by_label_status(dataset = dataset, labelled = True),
    }

    evaluation = Evaluation(
        dataset = dataset,
        subsets = subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    regr_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm', 'Rel_L2-norm', 'L1-norm', 'Rel_L1-norm']
    }

    ae_eval_metrics = {
        name: create_eval_metric(name)
        for name in ['L2-norm_reconstr', 'Rel_L2-norm_reconstr', 'L1-norm_reconstr', 'Rel_L1-norm_reconstr']
    }

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed')
   
    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        LossTermVisitor(loss_terms = regr_eval_metrics, eval_cfg = eval_cfg_comp),

        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        LossTermVisitor(loss_terms = ae_eval_metrics, eval_cfg = eval_cfg_reconstr),
    ]

    evaluation.accept_sequence(visitors = visitors)

    return evaluation


