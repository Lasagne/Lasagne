"""
Tools to train neural nets in Theano
"""

try:
    install_instr = """

Please make sure you install a recent enough version of Theano. Note that a
simple 'pip install theano' will usually give you a version that is too old
for Lasagne. See the installation docs for more details:
http://lasagne.readthedocs.org/en/latest/user/installation.html#theano"""
    import theano
except ImportError:  # pragma: no cover
    raise ImportError("Could not import Theano." + install_instr)
else:
    if not hasattr(theano.tensor.nnet, 'relu'):  # pragma: no cover
        raise ImportError("Your Theano version is too old." + install_instr)
    del install_instr
    del theano


from . import nonlinearities
from . import init
from . import layers
from . import objectives
from . import random
from . import regularization
from . import updates
from . import utils

try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("Lasagne").version
    del pkg_resources
except:
    pass
