"""
Tools to train neural nets in Theano
"""

try:
    import theano
except ImportError:  # pragma: no cover
    raise ImportError("""Could not import Theano.

Please make sure you install a recent enough version of Theano.  See
section 'Install from PyPI' in the installation docs for more details:
http://lasagne.readthedocs.org/en/latest/user/installation.html#install-from-pypi
""")
else:
    del theano

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
