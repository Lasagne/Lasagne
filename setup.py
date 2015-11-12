import os
import re
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    init_py = open(os.path.join(here, 'lasagne', '__init__.py')).read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'numpy',
    # 'Theano',  # we require a development version, see requirements.txt
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-pep8',
    ]

setup(
    name="Lasagne",
    version=version,
    description="A lightweight library to build and train neural networks "
                "in Theano",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Lasagne contributors",
    author_email="lasagne-users@googlegroups.com",
    url="https://github.com/Lasagne/Lasagne",
    license="MIT",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
