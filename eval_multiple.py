
import torch
import pandas as pd

from pathlib import Path

from evaluation import Evaluation
from evaluation.experiment_eval_funcs import evaluation_linear_regr, evaluation_dnn_regr, evaluation_ae_linear

from visualisation import *

from helper_tools import normalise_tensor, normalise_dataframe, dict_str
from helper_tools.results_hts import regression_metrics

from data_utils.obfuscation import DataObfuscator
from data_utils.info import identifier_col, time_col, ts_time_col, ts_ps_col, ts_cols, ts_rename_dict

from experiment_eval import extract_best_model_params

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



def plot_mae_histograms(evaluations: dict[str, Evaluation], output_keys: dict[str, str] = None, title: str = None, save_path: Path = None):

    mae_losses = {
        name: evaluation.results.losses['L1-norm']/2
        for name, evaluation in evaluations.items()
    }


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    multi_hist = TensorMultiHistogramPlot(
        data_dict = mae_losses,
        bins = 100,
        range = (0,1),
        xlabel = 'Absolute Error'
    )

    plot_matrix.add_plot_dict({
        (0,0): multi_hist,
    })

    plot_matrix.draw(fontsize = 14, figsize = (10, 8))





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

    
    results_dir = Path('./results_hyperopt/')

    experiment_names = [
        #'linear_regr_iso_key_raw', 
        'linear_regr_iso_key_min_max', 

        #'deep_NN_regr_key_raw',
        'deep_NN_regr_key_min_max',
   
        #'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max',

        # 'linear_regr_iso_max_raw', 
        # 'linear_regr_iso_max_min_max',

        # 'deep_NN_regr_max_raw', 
        # 'deep_NN_regr_max_min_max',

        # 'AE_linear_joint_epoch_max_raw',
        # 'AE_linear_joint_epoch_max_min_max',
    ]

    experiment_title_map = {
        'linear_regr_iso_key_min_max': r'Linear Model',
        'deep_NN_regr_key_min_max': r'DNN Model',
        'AE_linear_joint_epoch_key_min_max': r'AE-Linear Model',
    }

    experiment_evals: dict[str, Evaluation] = {}
    outputs_keys: dict[str, str] = {}
    for name in experiment_names:
        
        experiment_dir = results_dir / name
        title = experiment_title_map[name]

        data_kind = 'key' if 'key' in name else 'max'
        normaliser_kind = 'raw' if name.endswith('raw') else 'min_max'

        if name.startswith('linear'):
            eval_func = evaluation_linear_regr
            outputs_keys[title] = 'regr_iso'
        elif 'deep_NN' in name:
            eval_func = evaluation_dnn_regr
            outputs_keys[title] = 'regr_iso'
        elif 'AE' in name:
            eval_func = evaluation_ae_linear
            outputs_keys[title] = 'ae_regr'
        
        best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

        
        experiment_evals[title] = eval_func(
            model_dir = experiment_dir, 
            data_kind = data_kind, 
            normaliser_kind = normaliser_kind,
            best_model_params = best_model_params,
        )       

    
    eval_metrics = {name: evaluation.results.metrics for name, evaluation in experiment_evals.items()}

    regr_metrics = {
        name: calculate_metrics(evaluation=evaluation, outputs_key=outputs_keys[name])
        for name, evaluation in experiment_evals.items()
    }

    for name in experiment_evals.keys():
        print(
            f'Best entry metrics of {name}:\n'
            f'-------------------------------------\n'
            f'Eval metrics: \n{dict_str(eval_metrics[name])}\n'
            f'-------------------------------------\n'
            f'Regr metrics: \n{dict_str(regr_metrics[name])}\n'
            f'-------------------------------------\n'
        )
    

    ###--- Histogram ---###
    save_path = '../Presentation/assets/figures/results/aerror_hist_key_min_max.pdf'
    plot_mae_histograms(
        evaluations = experiment_evals,
        save_path = save_path,
    )

    


"""
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":
    
    eval_models()
    

    pass