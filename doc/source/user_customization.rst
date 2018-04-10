.. _user_customization:

3. User Customization
=====================

Implementing a new Model
------------------------

One of the standard use cases of ABCpy is to do inference of a probabilistic model that is not part of ABCpy. We now go
through the details of such a scenario using the (already implemented) Gaussian distribution to explain how to implement
such a model from scratch.

There are two main scenarios: First, we want to use our probabilistic model to explain a relationship between
*parameters* (considered random variables for inference) and *observed data*.  This is for example the case when we want
to do inference on mechanistic models that do not have a PDF. In this case, our model has to derive from
:py:class:`ProbabilisticModel <abcpy.probabilisticmodels.ProbabilisticModel>` and implement a few required methods as
for example :py:meth:`forward_simulate() <abcpy.probabilisticmodels.ProbabilisticModel.forward_simulate>`.

In the second scenario, we want to use the model to build a relationship between *different parameters* (between
different random variables for inference). Then our model is restricted to either output continuous or discrete
parameters. Consequently, the model must derive from either from :py:class:`Continuous
<abcpy.probabilisticmodels.Continuous>` or :py:class:`Discrete <abcpy.probabilisticmodels.Discrete>` and implement the
required methods. These two classes in turn derive from from :py:class:`ProbabilisticModel
<abcpy.probabilisticmodels.ProbabilisticModel>`, such that we have have to implement also the method required by the
latter class.

Let us go through the implementation of a simple Gaussian model. The model has to conform to the API specified by the
base class :py:class:`ProbabilisticModels <abcpy.probabilisticmodels.ProbabilisticModel>`, and thus must implement at
least the following methods:

* :py:meth:`ProbabilisticModels.__init__() <abcpy.probabilisticmodels.ProbabilisticModel.__init__>`
* :py:meth:`ProbabilisticModels._check_input() <abcpy.probabilisticmodels.ProbabilisticModel._check_input>`
* :py:meth:`ProbabilisticModels._check_output() <abcpy.probabilisticmodels.ProbabilisticModel._check_output>`
* :py:meth:`ProbabilisticModels.forward_simulate() <abcpy.probabilisticmodels.ProbabilisticModel.forward_simulate>`
* :py:meth:`ProbabilisticModels.get_output_dimension() <abcpy.probabilisticmodels.ProbabilisticModel.get_output_dimension>`

We want our model to work in both scenarios, so our model also has to conform to the API of :py:class:`Continuous
<abcpy.probabilisticmodels.Continuous>` since the model output, which is the resulting data from a forward simulation,
is from a continuous domain.

* :py:meth:`Continuous.pdf() <abcpy.probabilisticmodels.Continuous.pdf>`
* :py:meth:`Discrete.pmf() <abcpy.probabilisticmodels.Discrete.pmf>`

Initializing a New Model
^^^^^^^^^^^^^^^^^^^^^^^^

In the following we go through the implementation of a Gaussian generative model to explain the API in greater detail.
Since a Gaussian model generates continous numbers, the newly implement class derives from
:py:class:`abcpy.probabilisticmodels.Continuous` and the header look as follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
   :language: python
   :lines: 7

A good way to start implementing a new model is to define a convenient way to initialize it with its input parameters.
In ABCpy all input parameters (input models) of our Gaussian model are independent ProbabilisticModels (or
Hyperparameters) and should not be stored within the model we are going to write. We merely need a reference to the
input parameters or input models respectively. This reference is handled by the :py:class:`InputConnector
<abcpy.probabilisticmodels.InputConnector>` class. It is important that upon initialization of our model, we call the
init function ProbabilisticModels and pass an InputConnector object to it, as stated in the documentation:

.. autoclass::  abcpy.probabilisticmodels.ProbabilisticModel
   :members: __init__
   :noindex:

This leads to the following implementation:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 12-21
    :dedent: 4
    :linenos:

For convenience our init function expects a list of parameters :code:`[mu, sigma]`, where :code:`mu` is the mean and
:code:`\sigma` is the standard deviation which are the sole two parameters of our generative Gaussian model. We do some
basic syntactic checks on the input that throw exceptions if unreasonable input is provided. In line 9 we create in
InputConnector object from the factory method :py:meth:`from_list <abcpy.probabilisticmodels.InputConnector.from_list>`.
The resulting InputConnector creates links between our Gaussian model and the models (or hyperparameters) that are used
for mu and sigma at initialization time.

Additionally, every model instance should have a unique name, which should also be passed to the init function of the
super class.


Checking the Input
^^^^^^^^^^^^^^^^^^

The next function we implement is :py:meth:`_check_input <abcpy.probabilisticmodels.ProbabilisiticModels._check_input>`
which should behave as follows:

.. autoclass::  abcpy.probabilisticmodels.ProbabilisticModel
   :members: _check_input
   :noindex:

This leads to the following implementation:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 24-35
    :dedent: 4
    :linenos:


Checking the Output
^^^^^^^^^^^^^^^^^^^

We also need to check the output of the model. This method is commonly used in case our model is used as an input for
other models. When using an inference scheme that utilizes perturbation, the output of our model is slightly perturbed.
We have to make sure that the perturbed output is still valid for our model. Thus it is required to implement the method
:py:meth:`_check_output() <abcpy.probabilisticmodels.ProbabilisiticModels._check_output>` that should obey to the
following API

.. autoclass::  abcpy.probabilisticmodels.ProbabilisticModel
   :members: _check_output
   :noindex:

Since the output of a Gaussian generative model is a single number from the full real domain, we can restrict ourselves
to syntactic checks. However, one could easily image models for which the output it restricted to a certain domain. Then,
this function should return :code:`False` as soon as values are out of the desired domain.

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 38-43
    :dedent: 4
    :linenos:


Getting the Output Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In particular if our model is used as input for others models, we have expose the dimension of the produced output using
the following method:

.. autoclass::  abcpy.probabilisticmodels.ProbabilisticModel
   :members: get_output_dimension
   :noindex:

Since our model generates a single float number in one forward simulation, the implementation look the above function
is straight forward:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 46-47
    :dedent: 4
    :linenos:


Forward Simulation
^^^^^^^^^^^^^^^^^^

At the core of our model lies the capability to forward simulate the model to create pseudo observations. This method must
be implemented obeying the following API

.. autoclass::  abcpy.probabilisticmodels.ProbabilisticModel
   :members: forward_simulate
   :noindex:

A proper implementation look as follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 50-60
    :dedent: 4
    :linenos:

Note that both :code:`mu` and :code:`sigma` are stored in the list input values in the same order as we provided them
to the InputConnector object in the init function. Futher note that the output is a list of vectors, though the Gaussian
generative model only produces real numbers.


Calculating the Probability Density Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since our model also derives from :py:class:`Continuous <abcpy.probabilisticmodels.Continuous>` we also have to implement
a the following function that calculates the probability density function at specific point.

.. autoclass::  abcpy.probabilisticmodels.Continuous
   :members: pdf
   :noindex:

As mentioned above, this is only required if one wants to use our model as input for other models. An implementation looks
as follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 63-68
    :dedent: 4
    :linenos:

Our model now conforms to ABCpy and we can start inferring parameters in the
same way (see :ref:`Getting Started <gettingstarted>`) as we would do with shipped models. 

.. The complete example code can be found `here <https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/models/gaussian_python/normal_extended_model.py>`_


Wrap a Model Written in C++
---------------------------

There are several frameworks that help you integrating your C++/C code into
Python. We showcase examples for

* `Swig <http://www.swig.org/>`_
.. * `Pybind <https://github.com/pybind>`_

Using Swig
^^^^^^^^^^

Swig is a tool that creates a Python wrapper for our C++/C code using an
interface (file) that we have to specify. We can then import the wrapper and
in turn use your C++ code with ABCpy as if it was written in Python.

We go through a complete example to illustrate how to use a simple Gaussian
model written in C++ with ABCpy. First, have a look at our C++ model:

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.cpp
   :language: c++
   :lines: 9 - 17
   :linenos:

To use this code in Python, we need to specify exactly how to expose the C++
function to Python. Therefore, we write a Swig interface file that look as
follows:

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.i
   :language: c++
   :linenos:

In the first line we define the module name we later have to import in your
ABCpy Python code. Then, in curly brackets, we specify which libraries we want
to include and which function we want to expose through the wrapper.

Now comes the tricky part. The model class expects a method `forward_simulate` that
forward-simulates our model and which returns an array of syntetic
observations. However, C++/C does not know the concept of returning an array,
instead in C++/C we would provide a memory position (pointer) where to write
the results. Swig has to translate between the two concepts. We use actually an
Swig interface definition from numpy called `import_array`. The line

.. literalinclude:: ../../examples/extensions/models/gaussian_cpp/gaussian_model_simple.i
   :language: c++
   :lines: 18
   :linenos:

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
   :lines: 3 - 60
   :linenos:

The important lines are where we import the wrapper code as a module (line 3) and call
the respective model function (line 48).

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
    :linenos:

More complex R models are incorporated in the same way. To include this function
within ABCpy we include the following code at the beginning of our Python file:

.. literalinclude:: ../../examples/extensions/models/gaussian_R/gaussian_model.py
    :language: python
    :lines: 6 - 14
    :linenos:

This imports the R function `simple_gaussian` into the Python environment. We
need to build our own model to incorporate this R function as in the previous
section. The only difference is in the `forward_simulate` method of the class `Gaussian'.

.. literalinclude:: ../../examples/extensions/models/gaussian_R/gaussian_model.py
    :language: python
    :lines: 59
    :dedent: 8
    :linenos:

The default output for R functions in Python is a float vector. This must be
converted into a Python numpy array for the purposes of ABCpy.


Implementing a new Distance
---------------------------
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
--------------------------------------

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
