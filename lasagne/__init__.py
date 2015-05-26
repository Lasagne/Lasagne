"""
Tools to train neural nets in Theano
"""

from . import nonlinearities
from . import init
from . import layers
from . import objectives
from . import regularization
from . import updates
from . import utils


import pkg_resources
__version__ = pkg_resources.get_distribution("Lasagne").version
del pkg_resources
