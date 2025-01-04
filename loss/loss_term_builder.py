
from typing import Any
from functools import partial

from .loss_classes import *
from .adapters import *
from .decorators import *

from .determ_terms import *


class StandardLossTermBuilder:

    registered_loss_terms = {
        'L2-norm': {'cls': LpNorm, 'kwargs':{'p': 2}},
        'Rel_L2-norm': {'cls': RelativeLpNorm, 'kwargs':{'p': 2}},
        'L1-norm': {'cls': LpNorm, 'kwargs':{'p': 1}},
        'Rel_L1-norm': {'cls': RelativeLpNorm, 'kwargs':{'p': 1}},
        'Huber': {'cls': Huber, 'kwargs':{}},
        'Rel_Huber': {'cls': RelativeHuber, 'kwargs':{}},
    }

    def create_for_AE(self, name: str) -> LossTerm:
        
        selected_loss_term_spec = self.registered_loss_terms[name]
        loss_term = selected_loss_term_spec['cls'](**selected_loss_term_spec['kwargs'])
        
        return AEAdapter(loss_term)

    def create_for_Regr(self, name: str) -> Loss:
        
        selected_loss_term_spec = self.registered_loss_terms[name]
        loss_term = selected_loss_term_spec['cls'](**selected_loss_term_spec['kwargs'])

        return RegrAdapter(loss_term)
        
   