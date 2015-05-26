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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
