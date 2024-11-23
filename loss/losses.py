
import torch
from torch import Tensor
from .loss_classes import LossTerm, CompositeLossTerm
    
from .determ_terms import LpNorm, RelativeLpNorm, Huber, RelativeHuber

from loss.decorators import Weigh, Observe, Loss
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import AnalyticalKLDiv, MonteCarloKLDiv, GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import LogLikelihood, GaussianDiagLL, IndBetaLL, GaussianUnitVarLL



class VAELoss:

    def __init__(self, ll_cls: LogLikelihood = GaussianDiagLL, kld_cls: AnalyticalKLDiv | MonteCarloKLDiv = GaussianAnaKLDiv):

        self.ll_term = Weigh(ll_cls(), weight = -1)
        self.kld_term = kld_cls()

        self.clt = CompositeLossTerm({'Log-Likelihood': self.ll_term, 'KL-Divergence': self.kld_term})
        self.loss = Loss(self.clt)


    def __call__(self, **tensors: Tensor):
        return self.loss(**tensors)
