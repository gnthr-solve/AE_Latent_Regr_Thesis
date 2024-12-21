
import os
import ray
import logging

from ray import train, tune

from pathlib import Path

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

# os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"


"""
Optimise
-------------------------------------------------------------------------------------------------------------------------------------------
"""


if __name__=="__main__":

    from hyperoptim import run_experiment
    from hyperoptim.experiment_cfgs import deep_regr_cfg

    run_experiment(exp_cfg = deep_regr_cfg, max_concurrent = 2)