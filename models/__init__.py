
from .encoders import *
from .decoders import *
from .var_encoders import *
from .var_decoders import *

from .regressors import *

from .autoencoders import *
from .vae import *
from .naive_vae import *

from .composite import *


"""
In paper "Transformer-based hierarchical latent space VAE for interpretable remaining useful life prediction" variational encoder
latent distribution parameters are produced with different activations for mean and variance.

Since this is not possible in the current implementation, an idea could be to use a composed encoder,
where each part produces one of the parameters. 
Using Softplus for the variance parameter could avoid the exploding variance problem because we could predict the variance directly,
which showed better results in the Naive VAE implementation.
"""