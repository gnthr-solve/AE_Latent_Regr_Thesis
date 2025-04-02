###--- External Library Imports ---###
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from tqdm import tqdm


###--- Custom Imports ---###
from data_utils import DatasetBuilder, SplitSubsetFactory, retrieve_metadata
from data_utils.info import time_col, exclude_columns
from data_utils.data_filters import filter_by_machine

from models.regressors import LinearRegr, FunnelDNNRegr

from loss import (
    CompositeLossTerm,
    LpNorm,
    RelativeLpNorm,
    Huber,
    RelativeHuber,
    KMeansLoss,
)

from loss.clt_callbacks import LossTrajectoryObserver
from loss.decorators import Loss, Weigh, WeightedCompositeLoss, Observe
from loss.adapters import RegrAdapter

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    RegrOutputVisitor,
    LossTermVisitorS, LossTermVisitor,
    LossStatisticsVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools import dict_str

from visualisation.eval_plot_funcs import plot_3Dlatent_with_error, plot_3Dlatent_with_attribute
from visualisation.training_history_plots import plot_agg_training_losses



"""
Training Functions - Regressors Direct
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def train_linear_regr():
    """
    Train direct linear model. Plot regression weights.
    """
    ###--- Meta ---###
    epochs = 100
    batch_size = 20

    regr_lr = 1e-2
    scheduler_gamma = 0.99

    dataset_kind = 'key'

    observe_loss_dev = False
    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns
    )
    
    dataset = dataset_builder.build_dataset()
    

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")

    # Linear Regression
    regressor = LinearRegr(latent_dim = input_dim)

    
    ###--- Observation Test Setup ---###
    n_iterations_regr = len(dataloader_regr)
    dataset_size = len(regr_train_ds)


    ###--- Losses ---###
    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    if observe_loss_dev:

        loss_obs = LossTermObserver(
            n_epochs = epochs,
            dataset_size= dataset_size,
            batch_size= batch_size,
            name = 'Regr Loss',
            aggregated = True,
        )

        regr_loss = Loss(Observe(observer = loss_obs, loss_term = regr_loss_term))
    
    else:
        regr_loss = Loss(loss_term = regr_loss_term)

    regr_loss_test = Loss(regr_loss_term)

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            y_hat_batch = regressor(X_batch)

            loss_regr = regr_loss(
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_regr.backward()

            optimiser.step()


        scheduler.step()

    
    ###--- Plot Observations ---###
    if observe_loss_dev:
        plot_agg_training_losses(
            observed_losses = {'Loss': loss_obs.losses},
        )


    ###--- Test Loss ---###
    test_subsets = subset_factory.retrieve(kind = 'test')
    
    regr_test_ds = test_subsets['labelled']

    test_indices = regr_test_ds.indices
    X_test_l = dataset.X_data[test_indices]
    y_test_l = dataset.y_data[test_indices]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():
        y_test_l_hat = regressor(X_test_l)

        loss_regr = regr_loss_test(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    print(
        f"Regression Baseline:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )

    ###--- Experiment ---###
    import matplotlib.pyplot as plt

    #print(f'Regr weight shape: {regressor.regr_map.weight.shape}')
    weights_1, weights_2 = regressor.regr_map.weight.detach().unbind(dim = 0)
    bias_1, bias_2 = regressor.regr_map.bias.detach().unbind(dim = 0)

    col_indices = np.arange(1, dataset.X_dim)
    X_col_labels = dataset.alignm.retrieve_col_labels(indices = col_indices)
    y_col_labels = dataset.alignm.retrieve_col_labels(indices = [1,2], from_X = False)

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

        n_top_features = 20
        top_features_df = weight_df.head(n_top_features)

        # Plot the feature importance for the top 20 features
        plt.figure(figsize=(14, 6))
        plt.tight_layout()
        plt.barh(top_features_df['Feature'], top_features_df['Weight'], color='steelblue')
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {n_top_features} Feature Weights in Linear Model for {label}')
        plt.gca().invert_yaxis()  # Most important features on top
        plt.show()




def train_deep_regr():
    """
    Train direct DNN model.
    """
    ###--- Meta ---###
    epochs = 5
    batch_size = 30

    n_layers = 5
    activation = 'Softplus'

    regr_lr = 1e-2
    scheduler_gamma = 0.9

    dataset_kind = 'key'
    normaliser_kind = 'min_max'


    ###--- Dataset ---###
    normaliser = create_normaliser(normaliser_kind)

    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns
    )
    
    dataset = dataset_builder.build_dataset()
    

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset = dataset, train_size = 0.9)
    train_subsets = subset_factory.retrieve(kind = 'train')

    regr_train_ds = train_subsets['labelled']

    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)

    ###--- Models ---###
    input_dim = dataset.X_dim - 1
    print(f"Input_dim: {input_dim}")


    # Deep Regression
    regressor = FunnelDNNRegr(input_dim = input_dim, n_layers = n_layers, activation = activation)

    ###--- Observation Test Setup ---###
    n_iterations_regr = len(dataloader_regr)
    dataset_size = len(regr_train_ds)

    loss_obs = LossTermObserver(
        n_epochs = epochs,
        dataset_size= dataset_size,
        batch_size= batch_size,
        name = 'Regr Loss',
        aggregated = True,
    )


    ###--- Losses ---###
    regr_loss_term = RegrAdapter(Huber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeHuber(delta = 1))
    #regr_loss_term = RegrAdapter(RelativeLpNorm(p = 2))

    regr_loss = Loss(Observe(observer = loss_obs, loss_term = regr_loss_term))
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Training Loop Joint---###
    pbar = tqdm(range(epochs))

    for epoch in pbar:
        
        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            y_hat_batch = regressor(X_batch)

            loss_regr = regr_loss(
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_regr.backward()

            optimiser.step()


        scheduler.step()

    
    ###--- Plot Observations ---###
    plot_agg_training_losses(
        observed_losses = {'Loss': loss_obs.losses},
    )


    ###--- Test Loss ---###
    test_datasets = subset_factory.retrieve(kind = 'test')
    
    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_datasets,
        models = {'regressor': regressor},
    )

    eval_cfg = EvalConfig(data_key = 'labelled', output_name = 'regr_iso', mode = 'iso', loss_name = 'Huber')
    visitors = [
        RegrOutputVisitor(eval_cfg = eval_cfg),
        LossTermVisitorS(regr_loss_term, eval_cfg = eval_cfg),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_regr = results.metrics[eval_cfg.loss_name]
    print(
        f"Regression Baseline:\n"
        f"---------------------------------------------------------------\n"
        f"After {epochs} epochs with {len(dataloader_regr)} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )

