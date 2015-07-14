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

To install release 0.1 of Lasagne from PyPI, run the following
commands:

.. code-block:: bash

  pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/0.1/requirements.txt
  pip install Lasagne==0.1

Install from source
===================

Download and navigate to your copy of the Lasagne source. One way to obtain a
copy is by cloning the git repository, for example:

.. code-block:: bash

  git clone https://github.com/Lasagne/Lasagne.git

Once you have obtained a copy of the source, run the following command to
install Lasagne's dependencies:

.. code-block:: bash

  cd Lasagne
  pip install -r requirements.txt

To install the Lasagne package itself, run:

.. code-block:: bash

  python setup.py install  # or 'setup.py develop' if you're developing Lasagne itself
