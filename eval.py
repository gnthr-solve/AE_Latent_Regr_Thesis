
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
    CompositeLossTermObs,
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
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor, LossTermVisitor
)

from visualisation import *

from helper_tools import normalise_tensor
from helper_tools.setup import create_eval_metric, create_normaliser
from helper_tools.results_hts import regression_metrics

from visualisation.obfuscation import DataObfuscator
from data_utils.info import identifier_col, time_col, ts_time_col, ts_ps_col, ts_cols, ts_rename_dict


ADD_METRICS = ['Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm']
TRAINING_PARAMS = ['timestamp', 'time_total_s', 'epochs', 'batch_size', 'regr_lr', 'scheduler_gamma']


"""
Plotting Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_predicted_w_losses(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    ###--- Retrieve Matching Data ---###
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch

    y_batch = normalise_tensor(y_batch)
    y_hat_batch = normalise_tensor(y_hat_batch)


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    colored_scatter_pred_y = ColoredScatterPlot(
        x_data = y_hat_batch[:, 0],
        y_data = y_hat_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 'Upstack',
        y_label = 'Downstack',
        color_label = '$L_2$-error',
        #title = 'Predicted Values',
    )

    plot_matrix.add_plot_dict({
        (0,0): colored_scatter_pred_y,
    })

    plot_matrix.draw(fontsize = 12)



def plot_true_predicted_w_losses(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    ###--- Retrieve Matching Data ---###
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch

    y_batch = normalise_tensor(y_batch)
    y_hat_batch = normalise_tensor(y_hat_batch)


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    colored_scatter_true_y = ColoredScatterPlot(
        x_data = y_batch[:, 0],
        y_data = y_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 'Upstack',
        y_label = 'Downstack',
        color_label = '$L_2$-norm',
        title = 'True Values',
    )

    colored_scatter_pred_y = ColoredScatterPlot(
        x_data = y_hat_batch[:, 0],
        y_data = y_hat_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 'Upstack',
        y_label = 'Downstack',
        color_label = '$L_2$-norm',
        title = 'Predicted Values',
    )

    plot_matrix.add_plot_dict({
        (0,0): colored_scatter_pred_y,
        (1,0): colored_scatter_true_y,
    })

    plot_matrix.draw(fontsize = 12)



def plot_predicted_v_actual_separate(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    ###--- Retrieve Matching Data ---###
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch

    y_batch = normalise_tensor(y_batch)
    y_hat_batch = normalise_tensor(y_hat_batch)


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    colored_scatter_true_y = ColoredScatterPlot(
        x_data = y_batch[:, 0],
        y_data = y_hat_batch[:, 0],
        color_data = losses['L2-norm'],
        x_label = 'Upstack True',
        y_label = 'Upstack Predicted',
        color_label = '$L_2$-error',
        title = 'True vs. Predicted Upstack',
    )

    colored_scatter_pred_y = ColoredScatterPlot(
        x_data = y_batch[:, 1],
        y_data = y_hat_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 'Downstack True',
        y_label = 'Downstack Predicted',
        color_label = '$L_2$-error',
        title = 'True vs. Predicted Downstack',
    )

    plot_matrix.add_plot_dict({
        (0,0): colored_scatter_true_y,
        (1,0): colored_scatter_pred_y,
    })

    plot_matrix.draw(fontsize = 10)




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

    print(metrics)
    print(',\n'.join([f'{k}: {v}' for k, v in metrics_dict.items()]))




"""
Eval Specific Functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def dnn_regr_results(results_dir: Path, experiment_name: str, data_kind: str, normaliser_kind: str):

    ###--- Paths ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'
    model_paths = {model_path.stem: model_path for model_path in list(experiment_dir.glob("*.pt"))} 
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
    

    ###--- Load Results, identify best ---###
    results_df = pd.read_csv(results_path, low_memory = False)
    drop_cols = [col for col in TRAINING_PARAMS if col in results_df.columns]
    results_df.drop(columns = drop_cols, inplace = True)

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()
    best_result_values = {name: val for name, val in best_entry.items() if name == 'L2_norm' or name in ADD_METRICS}
    best_result_params = {name: val for name, val in best_entry.items() if name != 'L2_norm' and name not in ADD_METRICS}
    print(
        f'Best entry dict: \n{best_entry}\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best params: \n{best_result_params}\n'
        f'-------------------------------------\n'
    )

    regressor = DNNRegr(
        input_dim = input_dim,
        output_dim = 2,
        **best_result_params,
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

    #plot_true_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso')
    plot_predicted_v_actual_separate(evaluation=evaluation, outputs_key='regr_iso')
    calculate_metrics(evaluation=evaluation, outputs_key='regr_iso')



def linear_regr_results(results_dir: Path, experiment_name: str, data_kind: str, normaliser_kind: str):

    ###--- Paths ---###
    experiment_dir = results_dir / experiment_name
    
    results_path = experiment_dir / 'final_results.csv'
    model_paths = {model_path.stem: model_path for model_path in list(experiment_dir.glob("*.pt"))} 
    #print(model_paths)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = data_kind,
        normaliser = normaliser,
    )
    
    dataset = dataset_builder.build_dataset()
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")
    

    ###--- Load Results, identify best ---###
    results_df = pd.read_csv(results_path, low_memory = False)
    drop_cols = [col for col in TRAINING_PARAMS if col in results_df.columns]
    results_df.drop(columns = drop_cols, inplace = True)

    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()
    best_result_values = {name: val for name, val in best_entry.items() if name == 'L2_norm' or name in ADD_METRICS}
    best_result_params = {name: val for name, val in best_entry.items() if name != 'L2_norm' and name not in ADD_METRICS}
    print(
        f'Best entry dict: \n{best_entry}\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best params: \n{best_result_params}\n'
        f'-------------------------------------\n'
    )

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

    #title = 'Predictions on unn. key dataset'
    title = None
    save_path = experiment_dir / 'pred_v_actual.png'
    #plot_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso', title = title)
    #plot_predicted_v_actual_separate(evaluation=evaluation, outputs_key='regr_iso', title = title, save_path=save_path)
    calculate_metrics(evaluation=evaluation, outputs_key='regr_iso')

    ###--- Weights ---###
    weights_1, weights_2 = regressor.regr_map.weight.detach().unbind(dim = 0)
    bias_1, bias_2 = regressor.regr_map.bias.detach().unbind(dim = 0)

    col_indices = np.arange(1, dataset.X_dim)
    X_col_labels = dataset.alignm.retrieve_col_labels(indices = col_indices)
    y_col_labels = ['Upstack', 'Downstack']

    obfuscator = DataObfuscator(rename_dict = ts_rename_dict)
    X_col_labels = obfuscator.obfuscate(feature_names = X_col_labels)

    weight_df_1 = pd.DataFrame({'Feature': X_col_labels, 'Weight': weights_1.numpy()})
    weight_df_2 = pd.DataFrame({'Feature': X_col_labels, 'Weight': weights_2.numpy()})

    for label, weight_df, bias in zip(y_col_labels, [weight_df_1, weight_df_2], [bias_1, bias_2]):

        weight_df['Absolute Weight'] = weight_df['Weight'].abs()
        weight_df = weight_df.sort_values(by = 'Absolute Weight', ascending = False)
        print(
            f'{label}:\n'
            f'-------------------------------------\n'
            f'weights:\n{weight_df}\n'
            f'bias:\n{bias}\n'
            f'-------------------------------------\n'
        )

        weight_df.drop(columns = ['Absolute Weight'], inplace = True)
        weight_df.to_csv(experiment_dir/ f'regr_weights_{label}.csv', index = False)
        n_top_features = 15
        top_features_df = weight_df.head(n_top_features)

        # Plot the feature importance for the top 20 features
        plt.figure(figsize=(14, 8))
        plt.barh(top_features_df['Feature'], top_features_df['Weight'], color='steelblue', label = 'Feature', height = 0.5)
        plt.xlabel('Coefficient Value', fontsize = 14)
        plt.tick_params(axis='y', labelsize = 16)
        plt.tick_params(axis='x', labelsize = 12)
        plt.title(f'Top {n_top_features} Feature Weights on MAX for {label}', fontsize=14)
        #plt.title(f'Top {n_top_features} Feature Weights on KEY for {label}', fontsize=14)

        plt.tight_layout(pad=1.5)
        plt.gca().invert_yaxis()  # Most important features on top
        plt.grid(True, which = 'major')

        plt.savefig(
            experiment_dir/ f'top{n_top_features}_weights_{label}.png',
            dpi=200,
            bbox_inches='tight',
            pad_inches=0.5
        )
        
        plt.show()
        

"""
Call Specific Evals
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def eval_model():

    results_dir = Path('./results/')

    # data_kind = 'key'
    # #data_kind = 'max'
    # normaliser_kind = 'raw'
    # #normaliser_kind = 'min_max'

    # experiment_name = f'deep_NN_regr_{data_kind}_{normaliser_kind}'

    # dnn_regr_results(
    #     results_dir=results_dir, 
    #     experiment_name=experiment_name, 
    #     data_kind=data_kind, 
    #     normaliser_kind=normaliser_kind
    # )


    #data_kind = 'key'
    data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'
    experiment_name = f'linear_regr_iso_{data_kind}_{normaliser_kind}'

    linear_regr_results(
        results_dir=results_dir, 
        experiment_name=experiment_name, 
        data_kind=data_kind, 
        normaliser_kind=normaliser_kind
    )




"""
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    
    eval_model()


    pass