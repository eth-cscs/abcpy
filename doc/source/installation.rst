.. _installation:

1. Installation
===============

Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~~

Simplest way to install 
::
   pip3 install abcpy


Installation from Souce
~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to work on the source, clone the repository
::

   git clone https://github.com/eth-cscs/abcpy.git

Make sure all requirements are installed
::

   cd abcpy
   pip3 install -r requirements.txt

To create a package and install it do
::

   make package
   pip3 install build/dist/abcpy-0.4.0-py3-none-any.whl

Note that ABCpy requires Python3.



