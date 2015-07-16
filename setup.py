import os
from setuptools import find_packages
from setuptools import setup

version = '0.1.dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = ''
    # CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
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
    description="neural network tools for Theano",
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
    author="Sander Dieleman",
    author_email="sanderdieleman@gmail.com",
    url="https://github.com/Lasagne/Lasagne",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
