
import logging

from pathlib import Path

from hyperoptim import run_experiment
from hyperoptim.experiment_cfgs import (
    linear_regr_iso_cfg, 
    deep_NN_regr_cfg,
    shallow_NN_regr_cfg, 
    vae_iso_cfg, 
    ae_linear_joint_epoch_cfg,
    ae_deep_joint_epoch_cfg,
    nvae_linear_joint_epoch_cfg,
    nvae_deep_joint_epoch_cfg,

    ae_linear_joint_epoch_prime_cfg,
)


logger = logging.getLogger(__name__)


"""
Run Optimisation experiments
-------------------------------------------------------------------------------------------------------------------------------------------
Issues to address:
    Checkpoints store models with same name in created tmp directory.
    If two trial workers save their checkpoint at the same time, conflicts will occur and potentially the wrong models are persisted.
    Potentially change name.
"""
if __name__=="__main__":

    ###--- Workers, Save and Cleanup ---###
    save_frequency = 5
    cleanup_frequency = 10
    max_concurrent = 6
    should_resume = True
    replace_default_tmp = False
    restart_errored = True
    # only windows:
    max_retries = 5
    retry_delay = 10
    

    scheduler_kwargs = {
        'time_attr': 'training_iteration',
        'max_t': 200,
        'grace_period': 15,
        'reduction_factor': 2,
        'brackets': 2,
    }

    ###--- Set Up Log Config ---###
    results_dir = Path(f'./results/')
    logging.basicConfig(
        filename= results_dir / 'experiment.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        #level=logging.DEBUG,
    )


    ###--- Experiments to Run ---###
    experiment_cfgs = [
        linear_regr_iso_cfg, 
        #deep_NN_regr_cfg,
        #shallow_NN_regr_cfg,
        #vae_iso_cfg, 
        #ae_linear_joint_epoch_cfg,
        #nvae_linear_joint_epoch_cfg,
        #ae_deep_joint_epoch_cfg,
        #nvae_deep_joint_epoch_cfg,

        #ae_linear_joint_epoch_prime_cfg
    ]


    ###--- Run all Experiments in experiment_cfgs ---###
    for exp_cfg in experiment_cfgs:

        run_experiment(
            exp_cfg = exp_cfg, 
            save_frequency = save_frequency, 
            cleanup_frequency = cleanup_frequency, 
            max_concurrent = max_concurrent,
            should_resume = should_resume,
            scheduler_kwargs = scheduler_kwargs,
            replace_default_tmp = replace_default_tmp,
            restart_errored = restart_errored,
        )

        # run_experiment_windows(
        #     exp_cfg = exp_cfg, 
        #     save_frequency = save_frequency, 
        #     cleanup_frequency = cleanup_frequency, 
        #     max_concurrent = max_concurrent,
        #     should_resume = should_resume,
        #     max_retries = max_retries,
        #     retry_delay = retry_delay,
        #     replace_default_tmp = replace_default_tmp,
        #     restart_errored = restart_errored,
        # )
    