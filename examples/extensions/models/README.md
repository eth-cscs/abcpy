# Wrapping models written in external code

In this folder we showcase how to wrap models written in C++, R and FORTRAN. We use the same model in all cases (a simple gaussian one) and we also provide the corresponding Python implementation for the sake of reference.

## C++

We use [Swig](http://www.swig.org/) here to interface C++ with Python. In order to use that, an interface file has to be created correctly, which specifies how to interface C++ with Python. 

Check [here](https://abcpy.readthedocs.io/en/latest/user_customization.html#wrap-a-model-written-in-c) for more detailed explanation. 

### Instructions
 
1. Go inside the `gaussian_cpp` folder.
2. Run `make` (requires a C++ compiler, eg `g++`). This automatically creates an additional Python file (`gaussian_model_simple.py`) and a compiled file (`_gaussian_model_simple.so`).
3. Run the `pmcabc-gaussian_model_simple.py` file.


### Common issues

You may encounter some issue with the `boost` library which can be solved by installing it and putting it into the correct search path; in Ubuntu, install it with:

```sudo apt-get install  libboost-all-dev```

### Link Time Optimization (LTO):

For more efficient compilation, usually C++ compilers use LTO to link previously compiled libraries to the currently compiled code. That can lead to issues however in this case, if for instance the Python3 executable was compiled with another version of compiler than the one currently installed. For this reason, Makefile here disables LTO by adding the flag `-fno-lto` to the two lines calling the C++ compiler. 

In case your C++ code is large and compilation takes long, you can remove those flags, even if that may break the compilation for the reasons outlined above. 

Check [here](https://github.com/ContinuumIO/anaconda-issues/issues/6619) for more information.

## FORTRAN

We can use easily the [F2PY](https://numpy.org/doc/stable/f2py/) tool to connect FORTRAN code to Python. This is part of Numpy. 

### Instructions

1. Go inside the `gaussian_f90` folder.
2. Run `make`; (requires a FORTRAN compiler, eg `F90`); this will produce a compiled file.
3. Run the `pmcabc-gaussian_model_simple.py` file.

## R

We use here the `rpy2` Python package to import R code in Python.

Check [here](https://abcpy.readthedocs.io/en/latest/user_customization.html#wrap-a-model-written-in-r) for more detailed explanation.

### Instructions

This does not require any compilation, as R is not a compiled language. 

1. Go inside the `gaussian_R` folder.
2. Run the `pmcabc-gaussian_model_simple.py` file, which includes code to import the corresponding R code.
