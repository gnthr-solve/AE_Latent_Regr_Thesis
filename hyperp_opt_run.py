
import os
import ray
import logging

from ray import train, tune

from pathlib import Path

from hyperoptim import run_experiment
from hyperoptim.experiment_cfgs import (
    linear_regr_iso_cfg, 
    deep_regr_cfg, 
    vae_iso_cfg, 
    ae_linear_joint_epoch_cfg,
    ae_deep_joint_epoch_cfg,
)


# os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"


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


    ###--- Experiments to Run ---###
    experiment_cfgs = [
        #linear_regr_iso_cfg, 
        deep_regr_cfg, 
        #vae_iso_cfg, 
        #ae_linear_joint_epoch_cfg,
        #ae_deep_joint_epoch_cfg,
    ]


    ###--- Run Experiments ---###
    for exp_cfg in experiment_cfgs:

        run_experiment(
            exp_cfg = exp_cfg, 
            save_frequency = save_frequency, 
            cleanup_frequency = cleanup_frequency, 
            max_concurrent = max_concurrent,
            should_resume = should_resume,
            replace_default_tmp = replace_default_tmp,
        )

    