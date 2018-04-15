# ABCpy [![Documentation Status](https://readthedocs.org/projects/abcpy/badge/?version=latest)](http://abcpy.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/eth-cscs/abcpy.svg?branch=master)](https://travis-ci.org/eth-cscs/abcpy)

ABCpy is a scientific library written in Python for Bayesian uncertainty quantification in
absence of likelihood function, which parallelizes existing approaximate Bayesian computation (ABC) 
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

# Documentation
For more information, check out the

* [Tutorial](http://abcpy.readthedocs.io/en/latest/README.html) 
* [Examples](https://github.com/eth-cscs/abcpy/tree/master/examples) directory and
* [Reference](http://abcpy.readthedocs.io/en/latest/abcpy.html)

Further, we provide a
[collection of models](https://github.com/eth-cscs/abcpy-models) for which ABCpy
has been applied successfully. This is a good place to look at more complicated inference setups.

# Author 
ABCpy was written by [Ritabrata Dutta, Università della svizzera italiana](https://search.usi.ch/en/people/c4342228614d041dca7e2f67cbb996c9/dutta-ritabrata) 
and [Marcel Schoengens, CSCS, ETH Zurich](schoengens@cscs.ch), and we're actively developing it. Please feel free 
to submit any bugs or feature requests. We'd also love to hear about your experiences with ABCpy in general. Drop us an email!

We want to thank [Prof. Antonietta Mira, Università della svizzera italiana](https://search.usi.ch/en/people/f8960de6d60dd08a79b6c1eb20b7442b/Mira-Antonietta), and [Prof. Jukka-Pekka Onnela, Harvard University](https://www.hsph.harvard.edu/onnela-lab/) for helpful 
contributions and advice; Avinash Ummadisinghu and Nicole Widmern correspondingly for developning dynamic-MPI backend and 
making ABCpy suitbale for Graphical models; and CSCS (Swiss Super Computing Center) for their generous support.

## Citation

There is a paper in the proceedings of the 2017 PASC conference. We would appreciate a citation. 

```
@inproceedings{Dutta:2017:AUE:3093172.3093233,
 author = {Dutta, Ritabrata and Schoengens, Marcel and Onnela, Jukka-Pekka and Mira, Antonietta},
 title = {ABCpy: A User-Friendly, Extensible, and Parallel Library for Approximate Bayesian Computation},
 booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},
 series = {PASC '17},
 year = {2017},
 isbn = {978-1-4503-5062-4},
 location = {Lugano, Switzerland},
 pages = {8:1--8:9},
 articleno = {8},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3093172.3093233},
 doi = {10.1145/3093172.3093233},
 acmid = {3093233},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {ABC, Library, Parallel, Spark},
} 
```

## Other Refernces

Other publications related to the ABCpy package:

* Comparison of ABC alorithms from an HPC perspective

```
@article{Dutta2017arXivABCpy,
  title={ABCpy: A High-Performance Computing Perspective to Approximate Bayesian Computation},
  author={Dutta, Ritabrata and Schoengens, Marcel and Ummadisingu, Avinash and Onnela, Jukka-Pekka and Mira, Antonietta},
  journal={arXiv preprint arXiv:1711.04694},
  year={2017}
}
```

* Bayesian Inference of Spreading Processes on Networks 

```
@article{2017arXiv170908862D,
    title = "{Bayesian Inference of Spreading Processes on Networks}",
   author = {{Dutta}, R. and {Mira}, A. and {Onnela}, J.-P.},
  journal = {ArXiv e-prints arXiv:1709.08862},
     year = {2017}
}
```
* Bayesian parameter estimation of platelets deposition model (towrads personalized clinical test)

```
@article{2017arXiv171001054D,
   author = {{Dutta}, R. and {Chopard}, B. and {L{\"a}tt}, J. and {Dubois}, F. and 
	{Zouaoui Boudjeltia}, K. and {Mira}, A.},
    title = "{Parameter estimation of platelets deposition: Approximate Bayesian computation with high performance computing}",
  journal = {ArXiv e-prints arXiv:1710.01054},
     year = {2017}
}
```
* Bayesin inference of Dynamic Queueing Networks 

```
@article{2018arXiv180402526E,
   author = {{Ebert}, A. and {Dutta}, R. and {Wu}, P. and {Mengersen}, K. and 
	{Ruggeri}, F. and {Mira}, A.},
    title = "{Likelihood-Free Parameter Estimation for Dynamic Queueing Networks}",
  journal = {ArXiv e-prints arXiv:1804.02526},
     year = {2018}
}
```
* Bayesian Calibration of Force-fields

```
@article{2018arXiv180402742D,
   author = {{Dutta}, R. and {Faidon Brotzakis}, Z. and {Mira}, A.},
    title = "{Bayesian Calibration of Force-fields from Experimental Data: TIP4P Water}",
  journal = {ArXiv e-prints arXiv:1804.02742},
     year = {2018}
}

```


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

