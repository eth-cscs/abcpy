# ABCpy [![Documentation Status](https://readthedocs.org/projects/abcpy/badge/?version=latest)](http://abcpy.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/eth-cscs/abcpy.svg?branch=master)](https://travis-ci.org/eth-cscs/abcpy) [![codecov](https://codecov.io/gh/eth-cscs/abcpy/branch/master/graph/badge.svg)](https://codecov.io/gh/eth-cscs/abcpy) [![DOI](https://zenodo.org/badge/doi/10.1145/3093172.3093233.svg)](http://dx.doi.org/10.1145/3093172.3093233) [![GitHub license](https://img.shields.io/github/license/eth-cscs/abcpy.svg)](https://github.com/eth-cscs/abcpy/blob/master/LICENSE) [![PyPI version shields.io](https://img.shields.io/pypi/v/abcpy.svg)](https://pypi.python.org/pypi/abcpy/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/abcpy.svg)](https://pypi.python.org/pypi/abcpy/)  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/eth-cscs/abcpy/master?filepath=examples)

ABCpy is a scientific library written in Python for Bayesian uncertainty quantification in
absence of likelihood function, which parallelizes existing approximate Bayesian computation (ABC) 
algorithms and other likelihood-free inference schemes. 

# Content

ABCpy presently includes the following **ABC algorithms**:

* [RejectionABC](https://www.genetics.org/content/145/2/505)
* [PMCABC (Population Monte Carlo ABC)](https://www.annualreviews.org/doi/abs/10.1146/annurev-ecolsys-102209-144621)
* [SMCABC (Sequential Monte Carlo ABC)](https://link.springer.com/article/10.1007/s11222-011-9271-y)
* [RSMCABC (Replenishment SMC-ABC)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1541-0420.2010.01410.x)
* [APMCABC (Adaptive Population Monte Carlo ABC)](https://link.springer.com/article/10.1007/s00180-013-0428-3)
* [SABC (Simulated Annealing ABC)](https://link.springer.com/article/10.1007/s11222-014-9507-8)
* [ABCsubsim (ABC using subset simulation)](https://epubs.siam.org/doi/10.1137/130932831)

The above can be used with the following **distances**: 

* Euclidean Distance
* [Logistic Regression and Penalised Logistic Regression (classification accuracy)](https://link.springer.com/article/10.1007/s11222-017-9738-6)
* Divergences between datasets: 
  * [Wasserstein Distance](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12312)
  * [Sliced Wasserstein Distance](https://ieeexplore.ieee.org/abstract/document/9054735)
  * [Gamma Divergence](http://proceedings.mlr.press/v130/fujisawa21a/fujisawa21a.pdf)
  * [Kullback Leibler Divergence](http://proceedings.mlr.press/v84/jiang18a/jiang18a.pdf)
  * [Maximum Mean Discrepancy](http://proceedings.mlr.press/v51/park16.pdf)
  * [Energy Distance](https://arxiv.org/abs/1905.05884)
  * [Squared Hellinger Distance](https://arxiv.org/pdf/2006.14126.pdf)
  
Moreover, we provide the following methods for directly **approximating the likelihood functions**:
* [Bayesian Synthetic Likelihood](https://www.tandfonline.com/doi/abs/10.1080/10618600.2017.1302882?journalCode=ucgs20)
* [Semiparametric Bayesian Synthetic Likelihood](https://link.springer.com/article/10.1007/s11222-019-09904-x)
* [Penalised Logistic Regression for Ratio Estimation](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Likelihood-Free-Inference-by-Ratio-Estimation/10.1214/20-BA1238.full)

The above likelihood approximation methods can be used with the following samplers: 

* [PMC (Population Monte Carlo)](https://www.tandfonline.com/doi/abs/10.1198/106186004X12803)
* Metropolis-Hastings MCMC (Markov Chain Monte Carlo)

Additional **features** are:
* plotting utilities for the obtained posterior
* several methods for summary selection:
  * [Semi-automatic summary selection (with Neural networks)](http://proceedings.mlr.press/v97/wiqvist19a/wiqvist19a.pdf)
  * [summary selection using distance learning (with Neural networks)](https://link.springer.com/article/10.1007/s13571-019-00208-8)
  * [Sufficient statistics of exponential family approximating the likelihood (with Neural networks)](https://arxiv.org/abs/2012.10903)
* [Random Forest Model Selection Scheme](https://academic.oup.com/bioinformatics/article/32/6/859/1744513)


ABCpy addresses the needs of domain scientists and data
scientists by providing

* a fully modularized framework that is easy to use and easy to extend, 
* a quick way to integrate your generative model into the framework (from C++, R etc.) and
* a non-intrusive, user-friendly way to parallelize inference computations (for your laptop to clusters, supercomputers and AWS)
* an intuitive way to perform inference on hierarchical models or more generally on Bayesian networks

# Documentation
For more information, check out the

* [Youtube video](https://www.youtube.com/watch?v=cf2uNo0UEBs) presenting the library
* [Documentation](http://abcpy.readthedocs.io/en/v0.6.3) 
* [Examples](https://github.com/eth-cscs/abcpy/tree/v0.6.3/examples) directory and
* [Companion paper](https://www.jstatsoft.org/article/view/v100i07)


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
University](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/dutta/) and [Marcel Schoengens](mschoengens@bitvalve.org), CSCS, ETH Zurich, and presently actively maintained by [Lorenzo Pacchiardi, Oxford University](http://www.lorenzopacchiardi.me/) and [Ritabrata Dutta, Warwick
University](https://warwick.ac.uk/fac/sci/statistics/staff/academic-research/dutta/). Please feel free to submit any bugs or feature requests. We'd also love to hear about your experiences with ABCpy in general. Drop us an email!

We want to thank [Prof. Antonietta Mira, Università della svizzera
italiana](https://search.usi.ch/en/people/f8960de6d60dd08a79b6c1eb20b7442b/Mira-Antonietta),
and [Prof. Jukka-Pekka Onnela, Harvard
University](https://www.hsph.harvard.edu/onnela-lab/) for helpful contributions
and advice; Avinash Ummadisinghu and Nicole Widmern respectively for developing
dynamic-MPI backend and making ABCpy suitable for hierarchical models; and
finally CSCS (Swiss National Super Computing Center) for their generous support.

## Citation

There is a [paper](https://doi.org/10.18637/jss.v100.i07) in the _Journal of Statistical Software_. In case you use
ABCpy for your publication, we would appreciate a citation. You can use
[this](https://github.com/eth-cscs/abcpy/blob/master/doc/literature/JSS_2021.bib) BibTex reference.


## Other References

Publications in which ABCpy was applied:

* L. Pacchiardi, R. Dutta. "Generalized Bayesian Likelihood-Free Inference Using Scoring Rules Estimators", 2021, arXiv:2104.03889.

* L. Pacchiardi, R. Dutta. "Score Matched Conditional Exponential Families for Likelihood-Free Inference", 2022, Journal of Machine Learning Research 23(38):1−71.

* R. Dutta, K. Zouaoui-Boudjeltia, C. Kotsalos, A. Rousseau, D. Ribeiro de Sousa, J. M. Desmet, A. Van Meerhaeghe, A. Mira, and B. Chopard. "Interpretable pathological test for Cardio-vascular disease: Approximate Bayesian computation with distance learning.", 2020, arXiv:2010.06465.

* R. Dutta, S. Gomes, D. Kalise, L. Pacchiardi. "Using mobility data in the design of optimal lockdown strategies for the COVID-19 pandemic in England.", 2021,  PLOS Computational Biology, 17(8), e1009236.

* L. Pacchiardi, P. K&#252;nzli, M. Sch&#246;ngens, B. Chopard, R. Dutta, "Distance-Learning for Approximate Bayesian Computation to Model a Volcanic Eruption", 2021, Sankhya B, 83(1), 288-317.

* R. Dutta, J. P.  Onnela, A. Mira, "Bayesian Inference of Spreading Processes on Networks", 2018, Proceedings of Royal Society A, 474(2215), 20180129.

* R. Dutta, Z. Faidon Brotzakis and A. Mira, "Bayesian Calibration of   Force-fields from Experimental Data: TIP4P Water", 2018, Journal of Chemical Physics 149, 154110.
  
* R. Dutta, B. Chopard, J. Lätt, F. Dubois, K. Zouaoui Boudjeltia and A. Mira, "Parameter Estimation of Platelets Deposition: Approximate Bayesian Computation with High Performance Computing", 2018, Frontiers in physiology, 9.

* A. Ebert, R. Dutta, K. Mengersen, A. Mira, F. Ruggeri and P. Wu, "Likelihood-free parameter estimation for dynamic queueing networks: case study of passenger flow in an international airport terminal", 2021, Journal of Royal Statistical Society: Series C (Applied Statistics) 70.3: 770-792.

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

