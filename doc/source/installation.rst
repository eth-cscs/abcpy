.. _installation:

1. Installation
===============

ABCpy requires Python3 and is not compatible with Python2. The simplest way to install ABCpy is via PyPI and we
recommended to use this method.

Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

Simplest way to install 
::

   pip3 install abcpy

This also works in a virtual environment.


Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to work on the source, clone the repository
::

   git clone https://github.com/eth-cscs/abcpy.git

Make sure all requirements are installed
::

   cd abcpy
   pip3 install -r requirements.txt

To create a package and install it, do
::

   pip3 install wheel
   make package
   pip3 install build/dist/abcpy-0.6.3-py3-none-any.whl

``wheel`` is required to install in this way.


Note that ABCpy requires Python3.

Requirements
~~~~~~~~~~~~


Basic requirements are listed in ``requirements.txt`` in the repository (`click here
<https://github.com/eth-cscs/abcpy/blob/master/requirements.txt>`_). That also includes packages required for MPI parallelization there, which is very often used. However, we also provide support for parallelization with Apache Spark (see below).

Additional packages are required for additional features:


- ``torch`` is needed in order to use neural networks to learn summary statistics. It can be installed by running: ::

    pip install -r requirements/neural_networks_requirements.txt
- In order to use Apache Spark for parallelization, ``findspark`` and ``pyspark`` are required; install them by: ::

    pip install -r requirements/backend-spark.txt



Troubleshooting ``mpi4py`` installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mpi4py`` requires a working MPI implementation to be installed; check the `official docs
<https://mpi4py.readthedocs.io/en/stable/install.html>`_ for more info. On Ubuntu, that can be installed with:
::

    sudo apt-get install libopenmpi-dev

Even when that is present, running ``pip install mpi4py`` can sometimes lead to errors. In fact, as specified in the `official docs
<https://mpi4py.readthedocs.io/en/stable/install.html>`_, the ``mpicc`` compiler needs to be in the search path. If that is not the case, a workaround is:
::

    env MPICC=/path/to/mpicc pip install mpi4py

In some cases, even the above may not be enough. A possibility is using ``conda`` (``conda install mpi4py``) which usually handles package dependencies better than ``pip``. Alternatively, you can try by installing directly ``mpi4py`` from the package manager; in Ubuntu, you can do:
::

    sudo apt install python3-mpi4py

which however does not work with virtual environments.

