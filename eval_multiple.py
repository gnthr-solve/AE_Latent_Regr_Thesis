
import os
import tempfile
import torch
import pandas as pd
import logging

from pathlib import Path

from data_utils import TensorDataset, SplitSubsetFactory, DatasetBuilder, get_subset_by_label_status

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, DNNRegr
from models import AE, VAE, GaussVAE, EnRegrComposite
from models.naive_vae import NaiveVAE_LogVar, NaiveVAE_Sigma, NaiveVAE_LogSigma

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
)

from loss.decorators import Loss, Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import RegrOutputVisitor, LossTermVisitor

from visualisation import *

from helper_tools import normalise_tensor, normalise_dataframe, dict_str
from helper_tools.setup import create_eval_metric, create_normaliser
from helper_tools.results_hts import regression_metrics

from data_utils.obfuscation import DataObfuscator
from data_utils.info import identifier_col, time_col, ts_time_col, ts_ps_col, ts_cols, ts_rename_dict


ADD_METRICS = ['Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm']
TRAINING_PARAMS = ['timestamp', 'time_total_s', 'epochs', 'batch_size', 'regr_lr', 'scheduler_gamma']


"""
Plotting Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_multiple_predicted_w_losses(
        evaluations: dict[str, Evaluation], 
        outputs_key: str,
        subtitle_map: dict[str, str],
        plt_title: str = None,
        save_path: Path = None
    ):


    ###--- Plotting Matrix ---###
    if plt_title:
        plot_matrix = PlotMatrix(title=plt_title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}


    ###--- True Predictions ---###
    y_data_df = pd.read_csv('./data/y_data.csv', low_memory = False)

    notna_mask = y_data_df.notna().all(axis = 1)
    y_data_df = y_data_df[notna_mask]

    y_data_df.drop(columns=['mapping_idx'], inplace = True)
    y_data_df_normed_axis = normalise_dataframe(y_data_df, columns = ['MRR_up_mean', 'MRR_down_mean'])

    plot_dict[(0,0)] = DFScatterPlot(
        df = y_data_df_normed_axis, 
        x_col = 'MRR_up_mean', 
        y_col = 'MRR_down_mean',
        color = 'green',
        x_label = 'Upstack',
        y_label = 'Downstack',
        title = 'True values'
    )


    ###--- Evaluation Plots ---###
    for n, (name, evaluation) in enumerate(evaluations.items(), start = 1):

        i = n // 3
        j = n % 3

        results = evaluation.results
        losses = results.losses
        metrics = results.metrics

        ###--- Retrieve Matching Data ---###
        y_batch = evaluation.test_data['labelled']['y_batch']
        y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch

        y_batch = normalise_tensor(y_batch)
        y_hat_batch = normalise_tensor(y_hat_batch)

        plot_dict[(i,j)] = ColoredScatterPlot(
            x_data = y_hat_batch[:, 0],
            y_data = y_hat_batch[:, 1],
            color_data = losses['L2-norm'],
            x_label = 'Upstack',
            y_label = 'Downstack',
            color_label = 'Sample $L_2$-error',
            title = subtitle_map[name],
        )

    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 12, figsize = (16, 8))





"""
Calculate Metrics
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def calculate_metrics(evaluation: Evaluation, outputs_key: str, title: str = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    ###--- Retrieve Matching Data ---###
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch
    
    metrics_dict = regression_metrics(y_true = y_batch, y_pred = y_hat_batch)

    # print(metrics)
    # print(',\n'.join([f'{k}: {v}' for k, v in metrics_dict.items()]))
    return metrics_dict





    



"""
Call Specific Evals
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def eval_models():

    from experiment_eval import linear_regr_evaluation, dnn_regr_evaluation, ae_linear_evaluation, extract_best_model_params
    results_dir = Path('./results_hyperopt/')

    experiment_names = [
        'linear_regr_iso_key_raw', 
        'linear_regr_iso_key_min_max', 

        'deep_NN_regr_key_raw',
        'deep_NN_regr_key_min_max',
   
        'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max',

        # 'linear_regr_iso_max_raw', 
        # 'linear_regr_iso_max_min_max',

        # 'deep_NN_regr_max_raw', 
        # 'deep_NN_regr_max_min_max',

        # 'AE_linear_joint_epoch_max_raw',
        # 'AE_linear_joint_epoch_max_min_max',
    ]

    # experiment_title_map = {
    #     'deep_NN_regr_key_raw': r'Deep on KEY unn.',
    #     'deep_NN_regr_key_min_max': r'Deep on KEY Min-Max',
    #     'deep_NN_regr_max_raw': r'Deep on MAX unn.',
    #     'deep_NN_regr_max_min_max': r'Deep on MAX Min-Max',
    #     'shallow_NN_regr_key_raw': r'Shallow on KEY unn.',
    # }

    experiment_evals: dict[str, Evaluation] = {}
    for name in experiment_names:
        
        experiment_dir = results_dir / name

        data_kind = 'key' if 'key' in name else 'max'
        normaliser_kind = 'raw' if name.endswith('raw') else 'min_max'

        if name.startswith('linear'):
            eval_func = linear_regr_evaluation
        elif 'deep_NN' in name:
            eval_func = dnn_regr_evaluation
        elif 'AE' in name:
            eval_func = ae_linear_evaluation
        
        best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

        experiment_evals[name] = eval_func(
            model_dir = experiment_dir, 
            data_kind = data_kind, 
            normaliser_kind = normaliser_kind,
            best_model_params = best_model_params,
        )       

    
    eval_metrics = {name: evaluation.results.metrics for name, evaluation in experiment_evals.items()}

    regr_metrics = {
        name: calculate_metrics(evaluation=evaluation, outputs_key='regr_iso')
        for name, evaluation in experiment_evals.items()
    }

    for name in experiment_names:
        print(
            f'Best entry metrics of {name}:\n'
            f'-------------------------------------\n'
            f'Eval metrics: \n{dict_str(eval_metrics[name])}\n'
            f'-------------------------------------\n'
            f'Regr metrics: \n{dict_str(regr_metrics[name])}\n'
            f'-------------------------------------\n'
        )
    

    ###--- Scatter ---###
    # eval_metrics = {name: evaluation.results.metrics for name, evaluation in experiment_evals.items()}
    # metric = 'L2-norm'
    # title_map = {
    #     name: title_str + f' with mean $L_2={eval_metrics[name][metric]:.3f}$'
    #     for name, title_str in experiment_title_map.items()
    # }

    # save_path = '../Thesis/assets/figures/results/DNN/dnn_predictions_all.pdf'
    # plot_multiple_predicted_w_losses(
    #     evaluations = experiment_evals,
    #     outputs_key = 'regr_iso',
    #     subtitle_map = title_map,
    #     save_path = save_path,
    # )

    


"""
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":
    
    eval_models()
    

    pass