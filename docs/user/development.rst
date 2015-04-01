Development
===========

The ``Lasagne`` project is developed by Sander Dieleman and contributors via
GitHub. The development began in September 2014.

It is developed on GitHub: https://github.com/benanne/Lasagne

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


We suggest reading the issues page https://github.com/benanne/Lasagne/issues
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


Documentation
-------------

The documentation is generated with `Sphinx <http://sphinx-doc.org/latest/index.html>`_.
On Debian derivates it can be installed with

.. code:: bash

    $ sudo apt-get install python-sphinx

Sphinx makes use of `reStructured Text <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_

The documentation can be built with ``make html``.



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
