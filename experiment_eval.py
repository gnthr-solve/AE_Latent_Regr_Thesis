
import torch
import pandas as pd

from pathlib import Path

from data_utils import TensorDataset

from models.regressors import LinearRegr


from evaluation import Evaluation
from evaluation.experiment_eval_funcs import evaluation_linear_regr, evaluation_dnn_regr, evaluation_ae_linear, evaluation_ae_deep

from visualisation import *

from helper_tools import normalise_tensor
from helper_tools.results_hts import regression_metrics

from data_utils.obfuscation import DataObfuscator
from data_utils.info import identifier_col, time_col, ts_time_col, ts_ps_col, ts_cols, ts_rename_dict, wear_cols


ADD_METRICS = ['Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm']

METRICS = ['L2_norm','Rel_L2-norm','L1-norm','Rel_L2_norm','L1-Norm', 'Rel_L1-norm', 'L2_norm_reconstr']
TRAINING_PARAMS = [
    'timestamp', 
    'time_total_s', 
    'epochs', 
    'batch_size', 
    'regr_lr',
    'encoder_lr', 
    'decoder_lr', 
    'scheduler_gamma',
    'ete_regr_weight'
]

EVAL_METRICS = ['Rel_L2-norm','L1-norm']


def extract_best_model_params(experiment_dir: Path) -> dict[str, Any]:
    """
    Loads the final_results DataFrame produced by a complete hyperparameter optimisation run and
        - identifies the best result by L2-error
        - prints the result metrics, training hyperparameters and model parameters
        - returns the model parameters
    
    Args
    ----------
        experiment_dir: Path
            Directory of the hyperparameter optimisation results.
    
    Returns
    ----------
        best_model_params: dict
            Dictionary of best model hyperparameters.
    """
    ###--- Load hyperopt results and sort by L2 ---###
    results_path = experiment_dir / 'final_results.csv'
    results_df = pd.read_csv(results_path, low_memory = False)

    #--- Drop rows of trials that errored or were terminated by scheduler ---#
    if 'training_completed' in results_df.columns: # indicated by training_completed in later version
        results_df = results_df[results_df['training_completed']]
        results_df.drop(columns=['training_completed'], inplace=True)

    else: # indicated by NaN metrics in earlier trials
        results_df.dropna(axis = 0, how = 'any', inplace = True)

    #sort, select best entry, convert to dict
    best_entry = results_df.sort_values(by = 'L2_norm').iloc[0].to_dict()


    ###--- Extract Params ---###
    best_result_values = {name: val for name, val in best_entry.items() if name in METRICS}
    best_training_params = {name: val for name, val in best_entry.items() if name in TRAINING_PARAMS}
    best_model_params = {name: val for name, val in best_entry.items() if name not in set([*METRICS, *TRAINING_PARAMS])}
    

    ###--- Print best result parameters ---###
    print(
        f'Best entry:\n'
        f'-------------------------------------\n'
        f'Best metrics: \n{best_result_values}\n'
        f'-------------------------------------\n'
        f'Best training params: \n{best_training_params}\n'
        f'-------------------------------------\n'
        f'Best model params: \n{best_model_params}\n'
        f'-------------------------------------\n'
    )

    return best_model_params



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

    plot_matrix.draw(fontsize = 14, figsize = (10, 6))



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



def plot_actual_v_predicted_separate(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

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
        #title = 'True vs. Predicted Upstack',
    )

    colored_scatter_pred_y = ColoredScatterPlot(
        x_data = y_batch[:, 1],
        y_data = y_hat_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 'Downstack True',
        y_label = 'Downstack Predicted',
        color_label = '$L_2$-error',
        #title = 'True vs. Predicted Downstack',
    )

    plot_matrix.add_plot_dict({
        (0,0): colored_scatter_true_y,
        (1,0): colored_scatter_pred_y,
    })

    plot_matrix.draw(fontsize = 14, figsize = (10, 8))




def plot_latent_w_losses(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics


    Z_batch = evaluation.model_outputs[outputs_key].Z_batch


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)


    dim_red_scatter = DimReducedScatterPlot(
        feature_tensor = Z_batch,
        color_data = losses['L2-norm'],
        color_label = '$L_2$-error',
        #title = 'True vs. Predicted Upstack',
    )

    plot_matrix.add_plot_dict({
        (0,0): dim_red_scatter,
    })

    plot_matrix.draw(fontsize = 14, figsize = (10, 8))




def tsne_exp(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):
    from sklearn.manifold import TSNE

    dataset = evaluation.dataset
    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    #reverse mapping from int:str to str:int, obfuscate keys to align with obfuscated weights
    obfuscator = DataObfuscator(rename_dict = ts_rename_dict)
    X_col_map_reversed = {obfuscator.obfuscate_ind(v): k-1 for k, v in dataset.alignm.X_col_map.items()}

    # extract X_batch and Z_batch from eval
    X_batch = evaluation.test_data['labelled']['X_batch']
    Z_batch = evaluation.model_outputs[outputs_key].Z_batch
    
    # cols from top 10 features in linear on KEY
    # feature_cols = [
    #     'Carrier Rotation Polish mean deriv',
    #     'Platen Rotation Polish mean deriv',
    #     'Slurry Flow Rate std',
    #     'Slurry Flow Rate ptp',
    #     'Slurry Flow Rate mean',
    #     'Retainer Ring Pressure Set Value Start',
    #     'Retainer Ring Pressure Set Value Ramp',
    # ]

    # cols from top 10 features in linear on MAX
    feature_cols = [
        'Carrier Rotation Polish mean deriv',
        'Platen Rotation Polish mean deriv',
        'Slurry Flow Rate Polish max deriv',
        'PZ-5 Pressure Set Value Ramp',
        'Time Start ptp',
        'Platen Rotation Torque Polish mean',
        'Carrier Swing Torque Polish mean deriv',
    ]
    X_features = {
        feature_name: X_batch[:, X_col_map_reversed[feature_name]]
        for feature_name in feature_cols
    }

    #print(X_features)
    perplexity: float = 30.0
    n_iter: int = 1000
    random_state: int = 42

    tsne = TSNE(
        n_components = 2,
        perplexity = perplexity,
        n_iter = n_iter,
        random_state = random_state
    )
    
    Z2_batch = tsne.fit_transform(Z_batch)


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}
    plot_dict[(0,0)] = ColoredScatterPlot(
        x_data = Z2_batch[:, 0],
        y_data = Z2_batch[:, 1],
        color_data = losses['L2-norm'],
        x_label = 't-SNE x-dim',
        y_label = 't-SNE y-dim',
        color_label = '$L_2$-error',
    )

    for n, (name, X_feature) in enumerate(X_features.items(), 1):

        i = n // 4
        j = n % 4
        
        plot_dict[(i,j)] = ColoredScatterPlot(
            x_data = Z2_batch[:, 0],
            y_data = Z2_batch[:, 1],
            color_data = X_feature,
            x_label = 't-SNE x-dim',
            y_label = 't-SNE y-dim',
            color_label = name,
        )



    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 14, figsize = (16, 8))




def plot_latent_w_attribute(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    dataset = evaluation.dataset
    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    org_feature_cols = wear_cols

    #reverse mapping from int:str to str:int, obfuscate keys to align with obfuscated weights
    obfuscator = DataObfuscator(rename_dict = ts_rename_dict)
    X_col_map_reversed = {obfuscator.obfuscate_ind(v): k-1 for k, v in dataset.alignm.X_col_map.items()}

    # extract X_batch and Z_batch from eval
    X_batch = evaluation.test_data['labelled']['X_batch']
    Z_batch = evaluation.model_outputs[outputs_key].Z_batch

    
    feature_cols = obfuscator.obfuscate(org_feature_cols)
    X_features = {
        feature_name: X_batch[:, X_col_map_reversed[feature_name]]
        for feature_name in feature_cols
    }


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}
    # plot_dict[(0,0)] = ColoredScatterPlot(
    #     x_data = Z_batch[:, 0],
    #     y_data = Z_batch[:, 1],
    #     color_data = losses['L2-norm'],
    #     x_label = '$z_1$',
    #     y_label = '$z_2$',
    #     color_label = '$L_2$-error',
    #     #title = 'True vs. Predicted Downstack',
    # )

    for n, (name, X_feature) in enumerate(X_features.items(), 0):

        i = n // 2
        j = n % 2
        
        plot_dict[(i,j)] = ColoredScatterPlot(
            x_data = Z_batch[:, 0],
            y_data = Z_batch[:, 1],
            color_data = X_feature,
            x_label = '$z_1$',
            y_label = '$z_2$',
            color_label = name,
        )


    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 14, figsize = (14, 6))




def plot_latent_w_true_and_attribute(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    dataset = evaluation.dataset
    results = evaluation.results
    losses = results.losses
    metrics = results.metrics

    org_feature_cols = wear_cols

    #reverse mapping from int:str to str:int, obfuscate keys to align with obfuscated weights
    obfuscator = DataObfuscator(rename_dict = ts_rename_dict)
    X_col_map_reversed = {obfuscator.obfuscate_ind(v): k-1 for k, v in dataset.alignm.X_col_map.items()}

    # extract X_batch and Z_batch from eval
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_batch = normalise_tensor(y_batch)

    X_batch = evaluation.test_data['labelled']['X_batch']
    Z_batch = evaluation.model_outputs[outputs_key].Z_batch

    
    feature_cols = obfuscator.obfuscate(org_feature_cols)
    X_features = {
        feature_name: X_batch[:, X_col_map_reversed[feature_name]]
        for feature_name in feature_cols
    }


    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}
    plot_dict[(0,0)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data = y_batch[:, 0],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'True Upstack Values',
    )

    plot_dict[(0,1)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data =  y_batch[:, 1],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'True Downstack Values',
    )

    for n, (name, X_feature) in enumerate(X_features.items(), 2):

        i = n // 2
        j = n % 2
        
        plot_dict[(i,j)] = ColoredScatterPlot(
            x_data = Z_batch[:, 0],
            y_data = Z_batch[:, 1],
            color_data = X_feature,
            x_label = '$z_1$',
            y_label = '$z_2$',
            color_label = name,
        )



    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 14, figsize = (14, 6))





def plot_latent_w_true_and_pred(evaluation: Evaluation, outputs_key: str, title: str = None, save_path: Path = None):

    results = evaluation.results
    losses = results.losses
    metrics = results.metrics


    # extract X_batch and Z_batch from eval
    y_batch = evaluation.test_data['labelled']['y_batch']
    y_hat_batch = evaluation.model_outputs[outputs_key].y_hat_batch

    y_batch = normalise_tensor(y_batch)
    y_hat_batch = normalise_tensor(y_hat_batch)

    Z_batch = evaluation.model_outputs[outputs_key].Z_batch
    

    ###--- Plotting ---###
    if title:
        plot_matrix = PlotMatrix(title=title, save_path=save_path)
    else:
        plot_matrix = PlotMatrix(save_path=save_path)

    plot_dict = {}
    plot_dict[(0,0)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data = y_batch[:, 0],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'True Upstack Values',
    )

    plot_dict[(1,0)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data = y_batch[:, 1],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'True Downstack Values',
    )

    plot_dict[(0,1)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data = y_hat_batch[:, 0],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'Predicted Upstack Values',
    )

    plot_dict[(1,1)] = ColoredScatterPlot(
        x_data = Z_batch[:, 0],
        y_data = Z_batch[:, 1],
        color_data = y_hat_batch[:, 1],
        x_label = '$z_1$',
        y_label = '$z_2$',
        color_label = 'Predicted Downstack Values',
    )

    plot_matrix.add_plot_dict(plot_dict)

    plot_matrix.draw(fontsize = 14, figsize = (14, 6))





def plot_analyse_weights(regressor: LinearRegr, dataset: TensorDataset, experiment_dir: Path):
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
            #pad_inches=0.5
        )
        
        plt.show()





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
    return metrics_dict




"""
Call Specific Evals
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def eval_model_linear():

    results_dir = Path('./results_hyperopt/')

    #data_kind = 'key'
    data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'
    experiment_name = f'linear_regr_iso_{data_kind}_{normaliser_kind}'


    ###--- Get best model parameters and evaluate ---###
    experiment_dir = results_dir / experiment_name
    
    _ = extract_best_model_params(experiment_dir = experiment_dir)

    evaluation = evaluation_linear_regr(
        model_dir = experiment_dir,
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
    )


    ###--- Evaluation ---###
    calculate_metrics(evaluation=evaluation, outputs_key='regr_iso')

    # title = 'Predictions on unn. KEY dataset'
    # save_path = experiment_dir / 'pred_linear_key_unn.pdf'
    # plot_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso', title = title, save_path = save_path)


    title = None
    #title = 'Predictions on unn. KEY dataset'
    #save_path = '../Thesis/assets/figures/results/linear/true_v_pred_lin_KEY.pdf'
    save_path = '../Thesis/assets/figures/results/linear/true_v_pred_lin_MAX.pdf'
    plot_actual_v_predicted_separate(evaluation=evaluation, outputs_key='regr_iso', title = title, save_path=save_path)
    


def eval_model_dnn():

    results_dir = Path('./results_hyperopt/')

    data_kind = 'key'
    #data_kind = 'max'
    normaliser_kind = 'raw'
    #normaliser_kind = 'min_max'

    experiment_name = f'deep_NN_regr_{data_kind}_{normaliser_kind}'

    ###--- Get best model parameters and evaluate ---###
    experiment_dir = results_dir / experiment_name
    
    best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

    evaluation = evaluation_dnn_regr(
        model_dir=experiment_dir, 
        data_kind=data_kind, 
        normaliser_kind=normaliser_kind,
        best_model_params=best_model_params,
    )

    ###--- Evaluation ---###
    calculate_metrics(evaluation=evaluation, outputs_key='regr_iso')


    plot_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso')
    #plot_true_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso')
    #plot_actual_v_predicted_separate(evaluation=evaluation, outputs_key='regr_iso')




def eval_model_ae_linear():

    results_dir = Path('./results_hyperopt/')

    experiment_names = [
        'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max', 
        'AE_linear_joint_epoch_max_raw',
        'AE_linear_joint_epoch_max_min_max',
    ]

    #data_kind = 'key'
    data_kind = 'max'
    normaliser_kind = 'raw'
    #normaliser_kind = 'min_max'

    experiment_name = f'AE_linear_joint_epoch_{data_kind}_{normaliser_kind}'


    ###--- Get best model parameters and evaluate ---###
    experiment_dir = results_dir / experiment_name
    
    best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

    evaluation = evaluation_ae_linear(
        model_dir = experiment_dir, 
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
        best_model_params = best_model_params,
    )


    ###--- Calculate Metrics ---###
    calculate_metrics(evaluation=evaluation, outputs_key='ae_regr')


    ###--- Plot A ---###
    #save_path = '../Thesis/assets/figures/results/ae_regr/true_v_pred_lin_MAX.pdf'
    #plot_true_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso')


    ###--- Plot B ---###
    save_path = '../Thesis/assets/figures/results/ae_regr/true_v_pred_ae_lin_KEY_n.pdf'
    #save_path = '../Thesis/assets/figures/results/ae_regr/true_v_pred_ae_lin_MAX_n.pdf'
    #plot_actual_v_predicted_separate(evaluation=evaluation, outputs_key='ae_regr', save_path = save_path)


    ###--- Plot D ---###
    # #save_path = '../Thesis/assets/figures/results/ae_regr/tsne_KEY_features.pdf'
    # save_path = '../Thesis/assets/figures/results/ae_regr/tsne_MAX_features.pdf'
    # tsne_exp(evaluation=evaluation, outputs_key='ae_regr', save_path = save_path)




def eval_model_ae_linear_lim_dim():

    results_dir = Path('./results_hyperopt/')

    experiment_names = [
        'AE_linear_joint_epoch_l2_key_min_max',
        #'AE_linear_joint_epoch_l3_key_min_max', 
    ]

    data_kind = 'key'
    #data_kind = 'max'
    #normaliser_kind = 'raw'
    normaliser_kind = 'min_max'

    experiment_name = f'AE_linear_joint_epoch_l2_{data_kind}_{normaliser_kind}'


    ###--- Get best model parameters and evaluate ---###
    experiment_dir = results_dir / experiment_name
    
    best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

    evaluation = evaluation_ae_linear(
        model_dir = experiment_dir, 
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
        best_model_params = best_model_params,
    )


    ###--- Calculate Metrics ---###
    calculate_metrics(evaluation=evaluation, outputs_key='ae_regr')


    ###--- Plot A ---###
    #save_path = '../Thesis/assets/figures/results/ae_regr/latent2_ae_lin_KEY_n_consumables.pdf'
    save_path = None
    plot_latent_w_attribute(evaluation=evaluation, outputs_key='ae_regr', save_path=save_path)


    ###--- Plot B ---###
    # save_path = '../Thesis/assets/figures/results/ae_regr/latent_tvp_ae_lin_KEY_n.pdf'
    # plot_latent_w_true_and_pred(evaluation=evaluation, outputs_key='ae_regr', save_path = save_path)


    ###--- Plot B ---###
    # save_path = '../Thesis/assets/figures/results/ae_regr/latent2_ae_lin_KEY_n_consumables.pdf'
    # plot_latent_w_true_and_attribute(evaluation=evaluation, outputs_key='ae_regr', save_path = save_path)



    
def eval_model_ae_deep():

    results_dir = Path('./results_hyperopt/')

    experiment_names = [
        'AE_linear_joint_epoch_key_raw',
        'AE_linear_joint_epoch_key_min_max', 
        'AE_linear_joint_epoch_max_raw',
        'AE_deep_joint_epoch_key_raw',
    ]
    data_kind = 'key'
    #data_kind = 'max'
    normaliser_kind = 'raw'
    #normaliser_kind = 'min_max'

    experiment_name = f'AE_deep_joint_epoch_{data_kind}_{normaliser_kind}'


    ###--- Get best model parameters and evaluate ---###
    experiment_dir = results_dir / experiment_name
    
    best_model_params = extract_best_model_params(experiment_dir = experiment_dir)

    evaluation = evaluation_ae_deep(
        model_dir= experiment_dir, 
        data_kind = data_kind, 
        normaliser_kind = normaliser_kind,
        best_model_params = best_model_params,
    )

    ###--- Calculate Metrics ---###
    calculate_metrics(evaluation=evaluation, outputs_key='ae_regr')

    
    ###--- Plot A ---###
    #save_path = '../Thesis/assets/figures/results/linear/true_v_pred_lin_MAX.pdf'
    #plot_true_predicted_w_losses(evaluation=evaluation, outputs_key='regr_iso')


    ###--- Plot B ---###
    #save_path = '../Thesis/assets/figures/results/ae_regr/deep_true_v_pred_KEY_unn.pdf'
    #plot_actual_v_predicted_separate(evaluation=evaluation, outputs_key='ae_regr', save_path = save_path)


"""
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    #eval_model_linear()
    
    #eval_model_dnn()
    
    #eval_model_ae_linear()
    eval_model_ae_linear_lim_dim()
    #eval_model_ae_deep()

    pass