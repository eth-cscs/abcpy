# ABCpy 

ABCpy is a scientific library for approaximate Bayesian computation (ABC)
written in Python. It addresses the needs of domain scientists and data
scientists by providing

* a fully modularized framework that is easy to use and easy to extend, and
* a non-intrusive, user-friendly way to parallelize inference computations

## Main Features

* Quickly infer parameters for already existing models
* Quickly integrate your model into the framework
* Easily parallelize the inferrence computation when models become complex

## Getting Started
* [User Documentation](http://abcpy.readthedocs.io/en/latest/README.html)
* [Reference](http://abcpy.readthedocs.io/en/latest/abcpy.html)

.. Further, we provide a
.. [collection of models](https://github.com/eth-cscs/abcpy-models) for which ABCpy
.. has been applied successfully. This is a good place to look at more complicated inference setups.

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


In case of any questions, feel free to contact one of us:
* Ritabrata Dutta, University of Lugano
* Marcel Schoengens, CSCS, ETH Zurich

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

```
@article{Dutta2017arXivABCpy,
  title={ABCpy: A High-Performance Computing Perspective to Approximate Bayesian Computation},
  author={Dutta, Ritabrata and Schoengens, Marcel and Ummadisingu, Avinash and Onnela, Jukka-Pekka and Mira, Antonietta},
  journal={arXiv preprint arXiv:1711.04694},
  year={2017}
}
```

## Status
[![Documentation Status](https://readthedocs.org/projects/abcpy/badge/?version=latest)](http://abcpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/eth-cscs/abcpy.svg?branch=master)](https://travis-ci.org/eth-cscs/abcpy)
