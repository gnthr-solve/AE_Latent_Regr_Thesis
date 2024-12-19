
import os
import tempfile
import torch
import ray
import logging

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.train import Checkpoint, CheckpointConfig

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pathlib import Path

from data_utils import DatasetBuilder, SplitSubsetFactory

from preprocessing.normalisers import MinMaxNormaliser, MinMaxEpsNormaliser, ZScoreNormaliser, RobustScalingNormaliser

from models import (
    LinearEncoder,
    LinearDecoder,
    VarEncoder,
    VarDecoder,
)

from models.regressors import LinearRegr, ProductRegr, DNNRegr
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

from training.procedure_iso import AEIsoTrainingProcedure
from training.procedure_joint import JointEpochTrainingProcedure

from evaluation import Evaluation, EvalConfig
from evaluation.eval_visitors import (
    AEOutputVisitor, VAEOutputVisitor, RegrOutputVisitor,
    ReconstrLossVisitor, RegrLossVisitor,
)

from helper_tools.setup import create_normaliser
from helper_tools.ray_optim import custom_trial_dir_name, PeriodicSaveCallback, GlobalBestModelSaver

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

"""
Main Functions - Training
-------------------------------------------------------------------------------------------------------------------------------------------
"""

def AE_joint_epoch_procedure(config, dataset):
    
    #report_loss_name = 'Huber'
    report_loss_name = 'L2_norm'

    ###--- Meta ---###
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    
    n_layers = config['n_layers']
    activation = config['activation']

    encoder_lr = config['encoder_lr']
    decoder_lr = config['decoder_lr']
    regr_lr = config['regr_lr']
    scheduler_gamma = config['scheduler_gamma']

    ete_regr_weight = config['ete_regr_weight']

    ###--- Dataset Split ---###
    subset_factory = SplitSubsetFactory(dataset=dataset, train_size=0.9)
    train_subsets = subset_factory.retrieve(kind='train')
    
    ae_train_ds = train_subsets['unlabelled']
    regr_train_ds = train_subsets['labelled']


    ###--- DataLoader ---###
    dataloader_ae = DataLoader(ae_train_ds, batch_size = batch_size, shuffle = True)
    dataloader_regr = DataLoader(regr_train_ds, batch_size = batch_size, shuffle = True)


    ###--- Models ---###
    input_dim = dataset.X_dim - 1

    # encoder = LinearEncoder(input_dim = input_dim, latent_dim = latent_dim, n_layers = 4)
    # decoder = LinearDecoder(output_dim = input_dim, latent_dim = latent_dim, n_layers = 4)

    encoder = VarEncoder(input_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)
    decoder = VarDecoder(output_dim = input_dim, latent_dim = latent_dim, n_dist_params = 2, n_layers = n_layers, activation = activation)

    #model = NaiveVAE(encoder = encoder, decoder = decoder)
    ae_model = NaiveVAE_Sigma(encoder = encoder, decoder = decoder)
    
    #ae_model = AE(encoder = encoder, decoder = decoder)
    regressor = LinearRegr(latent_dim = latent_dim)
    #regressor = ProductRegr(latent_dim = latent_dim)


    ###--- Losses ---###
    reconstr_loss_term = AEAdapter(LpNorm(p = 2))
    #reconstr_loss_term = AEAdapter(RelativeLpNorm(p = 2))

    #regr_loss_term = RegrAdapter(Huber(delta = 1))
    regr_loss_term = RegrAdapter(LpNorm(p = 2))

    ete_loss_terms = {
        'Reconstruction Term': Weigh(reconstr_loss_term, weight = 1 - ete_regr_weight), 
        'Regression Term': Weigh(regr_loss_term, weight = ete_regr_weight),
    }

    ete_loss = Loss(CompositeLossTerm(ete_loss_terms))
    ae_loss = Loss(loss_term = reconstr_loss_term)
    

    ###--- Optimizer & Scheduler ---###
    optimiser = Adam([
        {'params': encoder.parameters(), 'lr': encoder_lr},
        {'params': decoder.parameters(), 'lr': decoder_lr},
        {'params': regressor.parameters(), 'lr': regr_lr},
    ])

    scheduler = ExponentialLR(optimiser, gamma = scheduler_gamma)


    ###--- Checkpoint condition ---###
    n_interim_checkpoints = 2
    epoch_modulo = epochs // n_interim_checkpoints
    checkpoint_condition = lambda epoch: (epoch % epoch_modulo == 0) or (epoch == epochs)


    ###--- Training Procedure ---###
    for epoch in range(epochs):
        
        ###--- Training Loop AE---###
        for iter_idx, (X_batch, _) in enumerate(dataloader_ae):
            
            X_batch = X_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)

            loss_vae = ae_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
            )

            #--- Backward Pass ---#
            loss_vae.backward()

            optimiser.step()


        ###--- Training Loop End-To-End ---###
        for iter_idx, (X_batch, y_batch) in enumerate(dataloader_regr):
            
            X_batch = X_batch[:, 1:]
            y_batch = y_batch[:, 1:]

            #--- Forward Pass ---#
            optimiser.zero_grad()
            
            Z_batch, X_hat_batch = ae_model(X_batch)
            y_hat_batch = regressor(Z_batch)

            loss_ete_weighted = ete_loss(
                X_batch = X_batch,
                X_hat_batch = X_hat_batch,
                y_batch = y_batch,
                y_hat_batch = y_hat_batch,
            )

            #--- Backward Pass ---#
            loss_ete_weighted.backward()

            optimiser.step()


        #--- Model Checkpoints & Report ---#
        if checkpoint_condition(epoch + 1):
            print(f'Checkpoint created at epoch {epoch + 1}/{epochs}')
            with tempfile.TemporaryDirectory() as tmp_dir:
                checkpoint = None
                #context = train.get_context()
                
                torch.save(ae_model.state_dict(), os.path.join(tmp_dir, "ae_model.pt"))
                torch.save(regressor.state_dict(), os.path.join(tmp_dir, f"regressor.pt"))


                checkpoint = Checkpoint.from_directory(tmp_dir)

                #NOTE: This reporting needs to be adjusted because the ETE loss is not the same as the regression loss
                train.report({report_loss_name: loss_ete_weighted.item()}, checkpoint=checkpoint)
        else:
            train.report({report_loss_name: loss_ete_weighted.item()})


        scheduler.step()


    ###--- Evaluation ---###
    test_subsets = subset_factory.retrieve(kind='test')

    evaluation = Evaluation(
        dataset = dataset,
        subsets = test_subsets,
        models = {'AE_model': ae_model,'regressor': regressor},
    )

    eval_cfg_reconstr = EvalConfig(data_key = 'unlabelled', output_name = 'ae_iso', mode = 'iso', loss_name = 'L2_norm_reconstr')
    eval_cfg_comp = EvalConfig(data_key = 'labelled', output_name = 'ae_regr', mode = 'composed', loss_name = report_loss_name)

    visitors = [
        AEOutputVisitor(eval_cfg = eval_cfg_reconstr),
        ReconstrLossVisitor(reconstr_loss_term, eval_cfg = eval_cfg_reconstr),

        AEOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrOutputVisitor(eval_cfg = eval_cfg_comp),
        RegrLossVisitor(regr_loss_term, eval_cfg = eval_cfg_comp),
    ]

    evaluation.accept_sequence(visitors = visitors)
    results = evaluation.results
    loss_reconstr = results.metrics[eval_cfg_reconstr.loss_name]
    loss_regr = results.metrics[eval_cfg_comp.loss_name]

    train.report({eval_cfg_comp.loss_name: loss_regr, eval_cfg_reconstr.loss_name: loss_reconstr})
    


"""
Optimise
-------------------------------------------------------------------------------------------------------------------------------------------
"""


if __name__=="__main__":

    ray.init()  # Initialize Ray

    ###--- Experiment Meta ---###
    experiment_name = 'AE_joint_epoch'
    #optim_metric = 'Huber'
    optim_metric = 'L2_norm'
    optim_mode = 'min'
    num_samples = 20

    storage_path = Path.cwd().parent / 'ray_results'

    #--- Dataset Meta ---#
    dataset_kind = 'key'
    normaliser_kind = 'min_max'
    exclude_columns = ["Time_ptp", "Time_ps1_ptp", "Time_ps5_ptp", "Time_ps9_ptp"]

    #--- Results Directory ---#
    dir_name = f'{experiment_name}_{dataset_kind}_{normaliser_kind}' if normaliser_kind != 'None' else f'{experiment_name}_{dataset_kind}'
    results_dir = Path(f'./results/{dir_name}/')
    os.makedirs(results_dir, exist_ok=True)


    ###--- Dataset Setup ---###
    normaliser = create_normaliser(normaliser_kind)
    dataset_builder = DatasetBuilder(
        kind = dataset_kind,
        normaliser = normaliser,
        exclude_columns = exclude_columns,
    )
    
    dataset = dataset_builder.build_dataset()


    ###--- Run Config ---###
    save_results_callback = PeriodicSaveCallback(
        save_frequency = 5, 
        experiment_name = experiment_name, 
        tracked_metrics=[optim_metric, 'L2_norm_reconstr'],
        results_dir=results_dir,
    )

    global_best_model_callback = GlobalBestModelSaver(
        tracked_metric = optim_metric,   
        mode = optim_mode,              
        cleanup_frequency = 10,       
        experiment_name = experiment_name,
        results_dir = results_dir,
    )

    checkpoint_cfg = CheckpointConfig(
        num_to_keep = 1, 
        checkpoint_score_attribute = optim_metric, 
        checkpoint_score_order = optim_mode
    )


    ###--- Searchspace ---###
    search_space = {
        'epochs': tune.randint(lower=2, upper = 20),
        'batch_size': tune.randint(lower=20, upper = 200),
        'latent_dim': tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'n_layers': tune.choice([3, 4, 5, 6, 7, 8, 9, 10]),
        'encoder_lr': tune.loguniform(1e-4, 1e-1),
        'decoder_lr': tune.loguniform(1e-4, 1e-1),
        'regr_lr': tune.loguniform(1e-4, 1e-1),
        'ete_regr_weight': tune.uniform(0, 1),
        'scheduler_gamma': tune.uniform(0.5, 1),
        'activation': tune.choice(['ReLU', 'LeakyReLU', 'PReLU', 'Softplus']),
    }

    
    ###--- Tune Config ---###
    #search_alg = BayesOptSearch()
    #search_alg = HyperOptSearch()
    search_alg = OptunaSearch()

    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)


    ###--- Setup and Run Optimisation ---###
    tuner = tune.Tuner(
        tune.with_parameters(AE_joint_epoch_procedure, dataset = dataset),
        tune_config=tune.TuneConfig(
            search_alg = search_alg,
            metric = optim_metric,
            mode = optim_mode,
            num_samples = num_samples,
            trial_dirname_creator = custom_trial_dir_name,
        ),
        run_config = train.RunConfig(
            name = experiment_name,
            storage_path = storage_path,
            checkpoint_config = checkpoint_cfg,
            callbacks = [save_results_callback, global_best_model_callback],
        ),
        param_space=search_space,
    )
    
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    
    results_df = results.get_dataframe()
    results_df.to_csv(results_dir / f'final_results.csv', index = False)