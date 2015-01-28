from __future__ import absolute_import
import os
from setuptools import find_packages
from setuptools import setup

version = '0.1dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = ''
    #CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'numpy',
    'Theano',
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

setup(
    name="Lasagne",
    version=version,
    description="neural network tools for Theano",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        ],
    keywords="",
    author="Sander Dieleman",
    author_email="sanderdieleman@gmail.com",
    url="https://github.com/benanne/lasagne",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
        },
    )
