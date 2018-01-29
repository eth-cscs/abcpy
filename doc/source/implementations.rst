.. _implementations:

3. User Customization
=====================

Implementing a new Model
~~~~~~~~~~~~~~~~~~~~~~~~

Often, one wants to use one of the provided inference schemes on a new
probabilistic model that is not part of ABCpy. We now go through the details of
such a scenario using the already implemented probabilistic model corresponding
to a Normal or Gaussian distribution to explain how to implement such a model
from scratch.

Every model has to conform to the API specified by the base class
:py:class:`abcpy.probabilisticmodels.ProbabilisticModel`. Thus, making a new
model compatible with ABCpy essentially boils down to implementing the following
methods:

.. literalinclude:: ../../abcpy/probabilisticmodels.py
    :language: python
    :lines: 4, 7, 130, 142, 175

However, these methods are the minimum that needs to be implemented in your
models builds a relationship between random variables and observed data. This is
for example the case when you want to do inference on mechanistic models that do
not have a PDF.

In case you want to use the model to create a relationship also between random
variables, two additional methods need to be implemented.

.. literalinclude:: ../../abcpy/probabilisticmodels.py
    :language: python
    :lines: 158, 193

To understand better the difference of both cases, please have a look at the
:ref:`Parameters as Random variables <implementations>` section.

Details of the Methods
----------------------

In the following we go through the required methods, explain what is expected,
and show how it would be implemented for the Gaussian model. It is always worth
consulting the reference for implementation details. For the constructor we
have:

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel.__init__
    :noindex:

The constructor expects to receive a list, containing all parameters of the new
model. These can be given in three ways:

1. A tupel, containing the parent, a
   :py:class:`abcpy.probabilisticmodels.ProbabilisticModel` object, as well as
   the output index. The output index refers to the index within a sample of the
   parent model which should be used for a parameter.

2. A probabilistic model object. This ensures, like the first point, that a
   graphical structure can be implemented.

3. A hyperparameter, which refers to any fixed value that can be given. The
   constructor of the base class is implemented such that fixed values (of any
   python type) will always be converted to a probabilistic model.

If we would like to implement our own constructor of a new model, we should in
the end call the constructor of the probabilistic model class. Consequently, we
would implement a simple version of a Gaussian model as follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 6-10

Note that we need to provide two **necessary** additional parameters that is not
required by the base class:


**self.name**. This is to provide the parameters with a name.  This is due to the fact that in the journal the final values for the random variables should be saved together with their name. However, since Python does not allow for easy retrieval of names given by the user, the name needs to be saved manually. We provide a default value such that the user does not need to specify such a name in case he wants to use this probabilistic model as a hierarchical model, which will not have its end value saved in the journal.

**self.dimension**: This attribute has to be defined for any probabilistic model you implement. It defines the dimension (length) a sample of your probabilistic model will have. Since a normal distribution will give one value per sample, its dimension is one. If we were to implement an n-dimensional multivariate normal distribution, the dimension would be n.

If you have a look at the definition of the constructor of the probabilistic model class, you might notice the following statement:

.. literalinclude:: ../../abcpy/probabilisticmodels.py
    :language: python
    :lines: 38
    :dedent: 8

Before this, all parameters given to the model are rewritten in the following way:

As we said before, each entry in the parameters list can be a probabilistic model, a fixed value or a tupel

However, for abcpy to work, all these different formats are rewritten to tupels during construction. If an n dimensional probabilistic model is given, the list you see denoted as *parents_temp* will contain n tupels, where the first entry is each time said probabilistic model and the second entry is numbered from 0 to n-1, the indices of a sampled value from the probabilistic model.

If a user provided a fixed value, this value is converted to an object of type :py:class:`abcpy.probabilisticmodels.Hyperparameter`, which derives from the probabilisic model class, and the tupel contains this object as the first entry and 0 as the second entry.

Finally, if a user used the access operator, the tupel will contain the probabilistic model as well as the index which was given in the access operator.

In pseudo-code, this list might look something like this:

.. code-block:: python

    [(prob_model_1, 0), (prob_model_1, 1), (prob_model_2, 2), (hyperparameter, 0)]

Within the constructor of :py:class:`abcpy.probablisticmodels.ProbabilisticModel`, the following method is called:

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel._check_parameters_at_initialization
    :noindex:

This method checks whether the parameters given at initialization are valid.

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 12-17
    :dedent: 4

This ensures that we give exactly two values to a the model and that the variance will not be smaller than 0.

Note that this method is not expected to have a return value. It is simply there to prevent the user from giving wrong inputs to probablistic models. If your model does not have any such constraints, you still need to implement the method, however, you can simply return without doing anything.

Next, we need the following method:

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel._check_parameters_before_sampling
    :noindex:


You might wonder what this method is for. We can imagine that our normal model might have a variance that is not a fixed value, but rather comes from some other probabilistic model. This so called parent might be able to sample negative values. Due to the graph structure in ABCpy, it would, therefore, be possible that our model would receive a negative value for its variance and for example would try to sample using that variance. This should not be possible.

So, we have the following implementation:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 19-22
    :dedent: 4

This method returns a boolean. It returns **True** if the parameters are accepted for sampling, and **False** otherwise.

Next, we need the method:

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel._check_parameters_fixed
    :noindex:

Again, let us explain the use of this method. A lot of the implemented ABC algorithms involve perturbing previously selected parameters using a perturbation kernel. Then, we try to fix the values for the parameters to these perturbed values. However, it could of course be possible that for some probabilistic model, the perturbed value is not acceptable. For example because the node can only return positive values, but the perturbation changed the parameter to some negative value. In this case, the parameters should be rejected.

However, for the normal model we are trying to implement, all values are acceptable. This is due to the fact that the range of a normal distribution is the real numbers.

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 24-25
    :dedent: 4

When implementing this method, keep in mind that it should decide whether the provided value or values can be sampled from this distribution.

Next, we get to the sampling method:

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel.sample_from_distribution
    :noindex:

Even if your model does not strictly implement a distribution, it is still named this way to avoid confusion. This method should simply sample from the distribution associated with the probabilistic model or simulate from a model.

Keep in mind that other methods will try to send their random number generator to this method during sampling. It is, therefore, recommended that you use a numpy random number generator, if you require one.

Also, even if you do not have any behavior implemented that requires a random number generator, it still needs to be passed to this function (due to the fact that other probabilistic models are based on random number generators). Hence, even if you do not need it, please specify the random number generator as a parameter.

Now, let's look at the implementation of the method for our model:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 27-38
    :dedent: 4

First, we need to obtain the values that correspond to each parameter of our model. Since the parents of our object can be probabilistic models, the values might not always be the same, and need to be obtained each time we want to sample. You do not need to implement the method used to to this, as long as you have derived your class from the probabilistic model class.

Now, we check whether the the values we obtained are okay to be used by our model. Whether this is the case forms the first entry in the list that we will return. Note that this is a necessary requirement. Other methods expect the first entry in this list to be a boolean corresponding to whether or not we could (and did) sample for this model.

Then, if the values are fine to be used, we sample using the random number generator, append this to the list that will be returned, and return the list.


Finally, we need to implement the probability density function.

.. automethod:: abcpy.probabilisticmodels.ProbabilisticModel.pdf
    :noindex:

**Again, this is not a must, but if there is no probability density function, it will only be possible to use the model as one of the hierarchical models (i.e. it cannot be part of the prior)**.

.. literalinclude:: ../../examples/extensions/models/gaussian_python/normal_extended_model.py
    :language: python
    :lines: 40-44

Again, we first need to obtain the values associated with all parents of the current model. However, we do not need to check these values, since pdfs will only be calculated after it is made sure that all values are allowed within the graph structure. We then calculate the pdf accordingly.

Our model now conforms to ABCpy and we can start inferring parameters in the
same way (see :ref:`Getting Started <gettingstarted>`) as we would do with shipped models. The
complete example code can be found `here
<https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/models/gaussian_python/normal_extended_model.py>`_


Wrap a Model Written in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several frameworks that help you integrating your C++/C code into
Python. We showcase examples for

* `Swig <http://www.swig.org/>`_
* `Pybind <https://github.com/pybind>`_

Using Swig
~~~~~~~~~~

Swig is a tool that creates a Python wrapper for our C++/C code using an
interface (file) that we have to specify. We can then import the wrapper and
in turn use your C++ code with ABCpy as if it was written in Python.

We go through a complete example to illustrate how to use a simple Gaussian
model written in C++ with ABCpy. First, have a look at our C++ model:

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.cpp
   :language: c++
   :lines: 9 - 17

To use this code in Python, we need to specify exactly how to expose the C++
function to Python. Therefore, we write a Swig interface file that look as
follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.i
   :language: c++

In the first line we define the module name we later have to import in your
ABCpy Python code. Then, in curly brackets, we specify which libraries we want
to include and which function we want to expose through the wrapper.

Now comes the tricky part. The model class expects a method `sample_from_distribution` that
forward-simulates our model and which returns an array of syntetic
observations. However, C++/C does not know the concept of returning an array,
instead in C++/C we would provide a memory position (pointer) where to write
the results. Swig has to translate between the two concepts. We use actually an
Swig interface definition from numpy called `import_array`. The line

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.i
   :language: c++
   :lines: 18

states that we want the two parameters `result` and `k` of the `gaussian_model`
C++ function be interpreted as an array of length k that is returned. Have a
look at the Python code below and observe how the wrapped Python function takes only two
instead of four parameters and returns a numpy array.

The first stop to get everything running is to translate the Swig interface file
to wrapper code in C++ and Python.
::

   swig -python -c++ -o gaussian_model_simple_wrap.cpp gaussian_model_simple.i

This creates two wrapper files `gaussian_model_simple_wrap.cpp` and
`gaussian_model_simple.py`. Now the C++ files can be compiled:
::

   g++ -fPIC -I /usr/include/python3.5m -c gaussian_model_simple.cpp -o gaussian_model_simple.o
   g++ -fPIC -I /usr/include/python3.5m -c gaussian_model_simple_wrap.cpp -o gaussian_model_simple_wrap.o
   g++ -shared gaussian_model_simple.o gaussian_model_simple_wrap.o -o _gaussian_model_simple.so

Note that the include paths might need to be adapted to your system. Finally, we
can write a Python model which uses our C++ code:

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/pmcabc-gaussian_model_simple.py
   :language: python
   :lines: 3 - 45

The important lines are where we import the wrapper code as a module (line 2) and call
the respective model function (line -2).

The full code is available in `examples/extensions/models/gaussion_cpp/`. To
simplify compilation of SWIG and C++ code we created a Makefile. Note that you
might need to adapt some paths in the Makefile.

Wrap a Model Written in R
-------------------------

Statisticians often use the R language to build statistical models. R models can
be incorporated within the ABCpy language with the `rpy2` Python package. We
show how to use the `rpy2` package to connect with a model written in R.

Continuing from the previous sections we use a simple Gaussian model as an
example. The following R code is the contents of the R file `gaussian_model.R`:

.. literalinclude:: ../../examples/extensions/models/gaussian_R/gaussian_model.R
    :language: R
    :lines: 1 - 4

More complex R models are incorporated in the same way. To include this function
within ABCpy we include the following code at the beginning of our Python file:

.. literalinclude:: ../../examples/extensions/models/gaussian_R/gaussian_model.py
    :language: python
    :lines: 6 - 14

This imports the R function `simple_gaussian` into the Python environment. We
need to build our own model to incorporate this R function as in the previous
section. The only difference is the `sample_from_distribution` method of the class `Gaussian'.

.. literalinclude:: ../../examples/extensions/models/gaussian_R/gaussian_model.py
    :language: python
    :lines: 65
    :dedent: 8

The default output for R functions in Python is a float vector. This must be
converted into a Python numpy array for the purposes of ABCpy.


Implementing a new Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

As discussed in the :ref:`Hierarchical model <gettingstarted>` section, our distance functions can, in general, act on multiple data sets. We provide a :py:class:`abcpy.distances.DefaultJointDistance` object to give a basic implementation of a distance acting on multiple data sets.

We will now explain how you can implement your own distance measure. A distance needs to provide the following three methods:

.. literalinclude:: ../../abcpy/distances.py
    :language: python
    :lines: 8, 14,29,66

Let us first look at the constructor. Distances in ABCpy should act on summary statistics. Therefore, a statistics calculator should be provided in the constructor. Also, since we want to implement a distance for multiple data sets, we need to provide a distance calculator that will act on a single data set. Note that you could also omit this and define the distance for one data set directly within your distance class. However, since we have already defined distances for single sets, we will use this here.

.. literalinclude:: ../../examples/extensions/distances/default_distance.py
    :language: python
    :lines: 16-18
    :dedent: 4

Then, we need to define how the total distance is calculated. In our case, we decide that we will iterate over each observed and simulated data set, calculate the individual distance between those, add all these individual distances, and in the end divide the result by the number of data sets that we calculated the distance of in this way.

.. literalinclude:: ../../examples/extensions/distances/default_distance.py
    :language: python
    :lines: 20-25
    :dedent: 4

Finally, we need to define the maximal distance that can be obtained from this distance measure. We then normalize the distance by dividing by the number of data sets.

.. literalinclude:: ../../examples/extensions/distances/default_distance.py
    :language: python
    :lines: 27-28
    :dedent: 4

The complete example for this tutorial can be found `here
<https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/distances/default_distance.py>`_.


Implementing a new Perturbation Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernels in ABCpy can be of two types. They can either be derived from the class :py:class:`abcpy.perturbationkernel.ContinuousKernel` or from :py:class:`abcpy.perturbationkernel.DiscreteKernel`. Whether it is a discrete or continuous kernel defines whether this kernel will act on discrete or continuous parameters (and, therefore, whether it has a probability mass or probability density function, respectively).

For this example, we will implement a kernel which perturbs continuous parameters using a multivariate normal distribution. This is one of the kernels already implemented within ABCpy.

A kernel always needs the following methods to be a valid object of type :py:class:`abcpy.perturbationkernel.PerturbationKernel`:

.. literalinclude:: ../../abcpy/perturbationkernel.py
    :language: python
    :lines: 8,11,21,40,60

First, we need to define a constructor.

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.__init__
    :noindex:


We expect that the arguments passed to the constructor should be of type :py:class:`abcpy.probabilisticmodels.ProbabilisticModel`, the random variables that should be perturbed using this kernel. All these models should be saved on the kernel for future reference.

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 5, 7,8

Next, we need the following method:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.calculate_cov
    :noindex:

This method calculates the covariance matrix for your kernel. Of course, not all kernels will have covariance matrices. However, since some kernels do, it is necessary to implement this method for all kernels. **If your kernel does not have a covariance matrix, simply return an empty list.**

The two arguments passed to this method are the accepted parameters manager and the kernel index. An object of type :py:class:`abcpy.acceptedparametersmanager.AcceptedParametersManager` is always initialized when an inference method object is instantiated. On this object, the accepted parameters, accepted weights, accepted covariance matrices for all kernels and other information is stored. This is such that various objects can access this information without much hassle centrally. To access any of the quantities mentioned above, you will have to call the `.value()` method of the corresponding quantity.

The second parameter, the kernel index, specifies the index of the kernel in the list of kernels that the inference method will in the end obtain. Since the user is expected to collect all his kernels in one object, this index will automatically be provided. You do not need any knowledge of what the index actually is. However, it is used to access the values relevant to your kernel, for example the current calculated covariance matrix for a kernel.

Let us now look at the implementation of the method:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 10, 25-30
    :dedent: 4

Some of the implemented inference algorithms weigh different sets of parameters differently. Therefore, if such weights are provided, we would like to weight the covariance matrix accordingly. We, therefore, check whether the accepted parameters manager contains any weights. If it does, we retrieve these weights, and calculate the covariance matrix using numpy, the parameters relevant to this kernel and the weights. If there are no weights, we simply calculate an unweighted covariance matrix.

Next, we need the method:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.update
    :noindex:


This method perturbs the parameters that are associated with the random variables the kernel should perturb.

The method again requires an accepted parameters manager and a kernel index. These have the same meaning as in the last method.


In addition to this, a row index is required, as well as a random number generator. The row index specifies which set of parameters should be perturbed. There are usually multiple sets, which should be perturbed by different workers during parallelization. You, again, need not worry about the actual value of this index.

The random number generator should be a random number generator compatible with numpy. This is due to the fact that other methods will pass their random number generator to this method, and all random number generators used within ABCpy are provided by numpy. Also, note that even if your kernel does not require a random number generator, you still need to pass this argument.

Here the implementation for our kernel:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 32, 53, 56-60
    :dedent: 4

The first line shows how you obtain the values of the parameters that your kernel should perturb. These values are converted to a numpy array. Then, the covariance matrix is retrieved from the accepted parameters manager using a similar function call. Finally, the parameters are perturbed and returned.

Last but not least, each kernel requires a probability density or probability mass function:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.pdf
    :noindex:

This method is implemented as follows for the multivariate normal:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 62, 83-87
    :dedent: 4

We simply obtain the parameter values and covariance matrix for this kernel and calculate the probability density function using scipy.

Note that after defining your own kernel, you will need to collect all your kernels in a :py:class:`abcpy.perturbationkernel.JointPerturbationKernel` object in order for inference to work. For an example on how to do this, check the :ref:`Using perturbation kernels <gettingstarted>` section.

The complete example used in this tutorial can be found `here
<https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/perturbationkernels/multivariate_normal_kernel.py>`_.
