
import torch

from .loss_classes import LossTerm, CompositeLossTerm
    
from .determ_terms import LpNorm, RelativeLpNorm, Huber, RelativeHuber

from loss.decorators import Weigh, Observe, Loss
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
    



class CompositeLossBuilder:

    def __init__(self, clt_name: str, observer_kwargs: dict):

        self.current_name = clt_name
        self.clt = CompositeLossTerm()
        self.observer_kwargs = observer_kwargs

    
    def add_term(self, name: str, loss_term: LossTerm, weight: float, observe: bool = False) -> LossTerm:

        loss_term = self._weigh_loss_term(loss_term = loss_term, weight = weight)

        if observe:
            loss_term = self._integrate_observer(name = name, loss_term = loss_term)
    
        self.clt = self.clt.add_term(name = name, loss_term = loss_term)


    def recurse(self, new_parent_name: str, weight: float, observe: bool = False):

        previous_clt = self.clt

        self.clt = CompositeLossTerm()
        self.add_term(name = self.current_name, loss_term = previous_clt, weight = weight, observe = observe)

        self.current_name = new_parent_name

        
    def build(self, observe: bool = False) -> LossTerm:

        clt = self.clt

        if observe:
            clt = self._integrate_observer(name = self.current_name, loss_term = clt)

        return clt
    

    def _weigh_loss_term(self, loss_term: LossTerm, weight: float) -> LossTerm:

        loss_term = Weigh(loss_term, weight)

        return loss_term

    
    def _integrate_observer(self, loss_term: LossTerm, name: str = None):

        observer = LossTermObserver(name = name, **self.observer_kwargs)
        loss_term = Observe(observer, loss_term)
        
        return loss_term