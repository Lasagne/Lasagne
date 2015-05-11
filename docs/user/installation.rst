.. _installation:

============
Installation
============

Lasagne currently requires Python 2.7 or 3.4 to run. Some of Lasagne's
dependencies such as ``numpy`` may require a C compiler to install.

For most installations, it is recommended to install Lasagne and its
dependencies inside a virtualenv or a conda environment. The
following commands assume that you have your environment active.

You can either install from PyPI or install from source, such as from
a Git clone.

Install from PyPI
=================

.. note:: Lasagne hasn't been released as of now.  The only way to
          install it for now is from source.

Simply run:

.. code-block:: bash

  pip install Lasagne

It's also a good practice to install dependencies with exactly the
same version numbers that the release was made with.  You can find the
``requirements.txt`` that defines those version numbers in the top
level directory of the Lasagne source tree.

Install from source
===================

Download and navigate to your copy of the Lasagne source. One way to obtain a
copy is by cloning the git repository, for example:

.. code-block:: bash

  git clone https://github.com/Lasagne/Lasagne.git

Once you have obtained a copy of the source, run the following command to
install its dependencies:

.. code-block:: bash

  pip install -r requirements.txt

To install the Lasagne package itself, run:

.. code-block:: bash

  python setup.py install  # or 'setup.py develop' if you're developing Lasagne itself
