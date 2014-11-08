.. _installation:

============
Installation
============

nntools currently requires Python 2.7 to run.  Some of nntools'
dependencies such as ``numpy`` may require a C compiler to install.

For most installations, it is recommended to install nntools and its
dependencies inside a virtualenv or a conda environment.  The
following commands assume that you have your environment active.

Install from PyPI
=================

Simply run:

.. code-block:: bash

  pip install nntools

It's also a good practice to install dependencies with exactly the
same version numbers that the release was made with.  You can find the
``requirements.txt`` that defines those version numbers in the top
level directory of the nntools source tree.

Install from source
===================

Download and navigate to your copy of the nntools source, then run:

.. code-block:: bash

  pip install -r requirements.txt

To install the nntools package itself, run:

.. code-block:: bash

  python setup.py install  # or 'setup.py develop' if you're developing nntools itself
