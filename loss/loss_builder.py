
import torch

from .loss_classes import Loss, LossTerm, CompositeLossTerm
    
from .determ_terms import LpNorm, RelativeLpNorm, Huber, RelativeHuber

from loss.decorators import Weigh, Observe
from loss.adapters import AEAdapter, RegrAdapter
from loss.vae_kld import GaussianAnaKLDiv, GaussianMCKLDiv
from loss.vae_ll import GaussianDiagLL, IndBetaLL, GaussianUnitVarLL

from observers import LossTermObserver, CompositeLossTermObserver, ModelObserver, VAELatentObserver

"""
Standard Loss Terms precomposed
-------------------------------------------------------------------------------------------------------------------------------------------
For now just for overview and potential refactor with builder or factory pattern
"""


"""
AE Loss Terms
-------------------------------------------------------------------------------------------------------------------------------------------
"""
l2_reconstr_term = AEAdapter(LpNorm(p = 2))
rel_l2_reconstr_term = AEAdapter(RelativeLpNorm(p = 2))



"""
VAE Loss Terms
-------------------------------------------------------------------------------------------------------------------------------------------
"""
gaussian_diag_ll_term = Weigh(GaussianDiagLL(), weight = -1)

gaussian_ana_kld_term = GaussianAnaKLDiv()
gaussian_MC_kld_term = GaussianMCKLDiv()

#loss_terms = {'Log-Likelihood': ll_term, 'KL-Divergence': kld_term}


class WeightedCompositeLossBuilder:

    
    def build(self, loss_terms: dict[str, LossTerm], weights: dict[str, float], observers: dict[str, LossTermObserver] = None) -> LossTerm:

        loss_terms = self._weigh_loss_terms(loss_terms=loss_terms, weights=weights)

        if observers is not None:
            loss_terms = self._register_observers(loss_terms=loss_terms, observers=observers)
    
        clt = CompositeLossTerm(**loss_terms)

        if observers is not None:
            clt = Observe(observers['composite'], clt)

        return clt
       

    def _weigh_loss_terms(self, loss_terms: dict[str, LossTerm], weights: dict[str, float]):

        loss_terms = {
            name: Weigh(loss_term, weights[name])
            for name, loss_term in loss_terms.items()
        }

        return loss_terms

    
    def _register_observers(self, loss_terms: dict[str, LossTerm], observers: dict[str, LossTermObserver]):

        loss_terms = {
            name: Observe(observers[name], loss_term)
            for name, loss_term in loss_terms.items()
            if observers.get(name, None) is not None
        }

        return loss_terms