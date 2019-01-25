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

This clearly works also in a virtual environment.


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

   make package

   pip3 install build/dist/abcpy-0.5.5-py3-none-any.whl


Note that ABCpy requires Python3.



