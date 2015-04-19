Development
===========

The ``Lasagne`` project is developed by Sander Dieleman and contributors via
GitHub. The development began in September 2014.

It is developed on GitHub: https://github.com/Lasagne/Lasagne

You can file issues and feature requests there.

Contributions
-------------

Everybody is welcome to contribute to ``Lasagne``. You can do so by

* Testing it and giving us feedback / opening issues on GitHub.

  * Writing unittests.

  * Simply using the software.

* Writing new code and sending it to us as pull requests. However, before you
  add new functionality you should eventually ask if we really want that as
  part of our project.

* Improving existing code.

* Suggesting something else how you can contribute.


We suggest reading the issues page https://github.com/Lasagne/Lasagne/issues
for more ideas how you can contribute.


Tools
-----

* ``pytests`` for unit testing

  * Install with: ``pip install pytest pytest-cov pytest-pep8``

* GitHub for hosting the source code
* http://lasagne.readthedocs.org/ for hosting the documentation


Code coverage can be tested with

.. code:: bash

    $ py.test

Testing will take over 5 minutes for the first run, but less than a minute for
subsequent runs when Theano can reuse compiled code.

You can add a so called pre-commit-hook to git to run all tests automatically
before you commit. The hook is installed by copying the following code to
`.git/hooks/pre-commit` and making this file executable:

.. code:: bash

    #!/usr/bin/env bash

    # Stash uncommited changes to make sure the commited ones will work
    git stash -q --keep-index
    # Run tests
    py.test
    RETURN_CODE=$?
    # Pop stashed changes
    git stash pop -q

    exit $RETURN_CODE


Documentation
-------------

The documentation is generated with `Sphinx <http://sphinx-doc.org/latest/index.html>`_.
On Debian derivates it can be installed with

.. code:: bash

    $ sudo apt-get install python-sphinx
    $ sudo -H pip install numpydoc

Sphinx makes use of `reStructured Text <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_

You should also `install pylearn2 <http://deeplearning.net/software/pylearn2/#download-and-installation>`_
to prevent warnings while generating the documentation.

The documentation can be built with ``make html``.

The documentation is written in numpydoc syntax. Information about numpydoc
can be found at the `numpydoc repository <https://github.com/numpy/numpydoc>`_,
especially `A Guide to NumPy/SciPy Documentation <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.



Project structure
-----------------

The project structure is

::

    .
    ├── docs
    │   ├── modules
    │   └── user
    ├── examples
    └── lasagne
        ├── layers
        ├── tests
        │   └── layers
        └── theano_extensions
