# ABCpy [![Documentation Status](https://readthedocs.org/projects/abcpy/badge/?version=latest)](http://abcpy.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/eth-cscs/abcpy.svg?branch=master)](https://travis-ci.org/eth-cscs/abcpy) [![codecov](https://codecov.io/gh/eth-cscs/abcpy/branch/master/graph/badge.svg)](https://codecov.io/gh/eth-cscs/abcpy) [![DOI](https://zenodo.org/badge/doi/10.1145/3093172.3093233.svg)](http://dx.doi.org/10.1145/3093172.3093233) [![GitHub license](https://img.shields.io/github/license/eth-cscs/abcpy.svg)](https://github.com/eth-cscs/abcpy/blob/master/LICENSE) [![PyPI version shields.io](https://img.shields.io/pypi/v/abcpy.svg)](https://pypi.python.org/pypi/abcpy/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/abcpy.svg)](https://pypi.python.org/pypi/abcpy/)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eth-cscs/abcpy/master?filepath=examples)

ABCpy is a scientific library written in Python for Bayesian uncertainty quantification in
absence of likelihood function, which parallelizes existing approximate Bayesian computation (ABC) 
algorithms and other likelihood-free inference schemes. It presently includes:

* RejectionABC
* PMCABC (Population Monte Carlo ABC)
* SMCABC (Sequential Monte Carlo ABC) 
* RSMCABC (Replenishment SMC-ABC)
* APMCABC (Adaptive Population Monte Carlo ABC)
* SABC (Simulated Annealing ABC)
* ABCsubsim (ABC using subset simulation)
* PMC (Population Monte Carlo) using approximations of likelihood functions
* Random Forest Model Selection Scheme
* Semi-automatic summary selection

ABCpy addresses the needs of domain scientists and data
scientists by providing

* a fully modularized framework that is easy to use and easy to extend, 
* a quick way to integrate your generative model into the framework (from C++, R etc.) and
* a non-intrusive, user-friendly way to parallelize inference computations (for your laptop to clusters, supercomputers and AWS)
* an intuitive way to perform inference on hierarchical models or more generally on Bayesian networks

# Documentation
For more information, check out the

* [Documentation](http://abcpy.readthedocs.io/en/v0.5.6) 
* [Examples](https://github.com/eth-cscs/abcpy/tree/v0.5.6/examples) directory and
* [Reference](http://abcpy.readthedocs.io/en/v0.5.6/abcpy.html)


Further, we provide a
[collection of models](https://github.com/eth-cscs/abcpy-models) for which ABCpy
has been applied successfully. This is a good place to look at more complicated inference setups.

# Quick installation and requirements


ABCpy can be installed from `pip`: 

    pip install abcpy

Check [here](https://abcpy.readthedocs.io/en/latest/installation.html) for more details.

Basic requirements are listed in `requirements.txt`. That also includes packages required for MPI parallelization there, which is very often used. However, we also provide support for parallelization with Apache Spark (see below).
 
 Additional packages are required for additional features: 

- `torch` is needed in order to use neural networks to learn summary statistics. It can be installed by running `pip install -r requirements/neural_networks_requirements.txt`
- In order to use Apache Spark for parallelization, `findspark` and `pyspark` are required; install them by `pip install -r requirements/backend-spark.txt`  

## Troubleshooting `mpi4py` installation

`mpi4py` requires a working MPI implementation to be installed; check the [official docs]((https://mpi4py.readthedocs.io/en/stable/install.html)) for more info. On Ubuntu, that can be installed with:

    sudo apt-get install libopenmpi-dev

Even when that is present, running `pip install mpi4py` can sometimes lead to errors. In fact, as specified in the [official docs]((https://mpi4py.readthedocs.io/en/stable/install.html)), the `mpicc` compiler needs to be in the search path. If that is not the case, a workaround is: 

    env MPICC=/path/to/mpicc pip install mpi4py

In some cases, even the above may not be enough. A possibility is using `conda` (`conda install mpi4py`) which usually handles package dependencies better than `pip`. Alternatively, you can try by installing directly `mpi4py` from the package manager; in Ubuntu, you can do:

    sudo apt install python3-mpi4py 

which however does not work with virtual environments.


# Author 
ABCpy was written by [Ritabrata Dutta, Warwick
University](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/dutta/) and [Marcel Schoengens](mschoengens@bitvalve.org), CSCS, ETH Zurich, and presently actively maintained by [Lorenzo Pacchiardi, Oxford University](https://github.com/LoryPack) and [Ritabrata Dutta, Warwick
University](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/dutta/). Please feel free to submit any bugs or feature requests. We'd also love to hear about your experiences with ABCpy in general. Drop us an email!

We want to thank [Prof. Antonietta Mira, Università della svizzera
italiana](https://search.usi.ch/en/people/f8960de6d60dd08a79b6c1eb20b7442b/Mira-Antonietta),
and [Prof. Jukka-Pekka Onnela, Harvard
University](https://www.hsph.harvard.edu/onnela-lab/) for helpful contributions
and advice; Avinash Ummadisinghu and Nicole Widmern respectively for developing
dynamic-MPI backend and making ABCpy suitable for hierarchical models; and
finally CSCS (Swiss National Super Computing Center) for their generous support.

## Citation

There is a [paper](http://dx.doi.org/10.1145/3093172.3093233) in the proceedings of the 2017 PASC conference. In case you use
ABCpy for your publication, we would appreciate a citation. You can use
[this](https://github.com/eth-cscs/abcpy/blob/v0.5.6/doc/literature/DuttaS-ABCpy-PASC-2017.bib) BibTex reference.


## Other References

Publications in which ABCpy was applied:

* L. Pacchiardi, R. Dutta. "Score Matched Conditional Exponential Families for Likelihood-Free Inference", 2020, arXiv:2012.10903.

* R. Dutta, K. Zouaoui-Boudjeltia, C. Kotsalos, A. Rousseau, D. Ribeiro de Sousa, J. M. Desmet, 
A. Van Meerhaeghe, A. Mira, and B. Chopard. "Interpretable pathological test for Cardio-vascular 
disease: Approximate Bayesian computation with distance learning.", 2020, arXiv:2010.06465.

* R. Dutta, S. Gomes, D. Kalise, L. Pacchiardi. "Using mobility data in the design of optimal 
lockdown strategies for the COVID-19 pandemic in England.", 2020, arXiv:2006.16059.

* L. Pacchiardi, P. K&#252;nzli, M. Sch&#246;ngens, B. Chopard, R. Dutta, "Distance-Learning for 
Approximate Bayesian Computation to Model a Volcanic Eruption", 2020, Sankhya B, ISSN 0976-8394, 
  [DOI: 10.1007/s13571-019-00208-8](https://doi.org/10.1007/s13571-019-00208-8).

* R. Dutta, J. P.  Onnela, A. Mira, "Bayesian Inference of Spreading Processes
  on Networks", 2018, Proc. R. Soc. A, 474(2215), 20180129.

* R. Dutta, Z. Faidon Brotzakis and A. Mira, "Bayesian Calibration of
  Force-fields from Experimental Data: TIP4P Water", 2018, Journal of Chemical Physics 149, 154110.
  
* R. Dutta, B. Chopard, J. Lätt, F. Dubois, K. Zouaoui Boudjeltia and A. Mira,
  "Parameter Estimation of Platelets Deposition: Approximate Bayesian
  Computation with High Performance Computing", 2018, Frontiers in physiology, 9.

* A. Ebert, R. Dutta, P. Wu, K. Mengersen and A. Mira, "Likelihood-Free
  Parameter Estimation for Dynamic Queueing Networks", 2018, arXiv:1804.02526.

* R. Dutta, M. Schoengens, L. Pacchiardi, A. Ummadisingu, N. Widerman, J. P.  Onnela, A. Mira, "ABCpy:       A High-Performance Computing Perspective to Approximate Bayesian Computation", 2020, arXiv:1711.04694.

## License
ABCpy is published under the BSD 3-clause license, see [here](LICENSE).

## Contribute
You are very welcome to contribute to ABCpy. 

If you want to contribute code, there are a few things to consider:
* a good start is to fork the repository
* know our [branching strategy](http://nvie.com/posts/a-successful-git-branching-model/)
* use GitHub pull requests to merge your contribution
* consider documenting your code according to the [NumPy documentation style guide](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
* consider writing reasonable [unit tests](https://docs.python.org/3.5/library/unittest.html)

