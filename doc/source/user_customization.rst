.. _user_customization:

3. User Customization
=====================

Implementing a new Model
------------------------

One of the standard use cases of ABCpy is to do inference on a probabilistic model that is not part of ABCpy. We now go
through the details of such a scenario using the (already implemented) Gaussian generative model to explain how to
implement it from scratch.

There are two scenarios to use a model: First, we want to use our probabilistic model to explain a relationship between
parameters* (considered random variables for inference) and *observed data*.  This is for example the case when we want
to do inference on mechanistic models that do not have a PDF. In this case, our model implementation has to derive from
:py:class:`ProbabilisticModel <abcpy.probabilisticmodels.ProbabilisticModel>` and a few abstract methods have to be
defined, as for example :py:meth:`forward_simulate() <abcpy.probabilisticmodels.ProbabilisticModel.forward_simulate>`.

In the second scenario, we want to use the model to build a relationship between *different parameters* (between
different random variables). Then our model is restricted to either output continuous or discrete parameters in form of
a vector. Consequently, the model must derive from either from :py:class:`Continuous
<abcpy.probabilisticmodels.Continuous>` or :py:class:`Discrete <abcpy.probabilisticmodels.Discrete>` and implement the
required abstract methods. These two classes in turn derive from from :py:class:`ProbabilisticModel
<abcpy.probabilisticmodels.ProbabilisticModel>`, such that the second scenario essentially extends the first.

Let us go through the implementation of a the Gaussian generative model. The model has to conform to the API specified
by the base class :py:class:`ProbabilisticModels <abcpy.probabilisticmodels.ProbabilisticModel>`, and thus must
implement at least the following methods:

* :py:meth:`ProbabilisticModels.__init__() <abcpy.probabilisticmodels.ProbabilisticModel.__init__>`
* :py:meth:`ProbabilisticModels._check_input() <abcpy.probabilisticmodels.ProbabilisticModel._check_input>`
* :py:meth:`ProbabilisticModels._check_output() <abcpy.probabilisticmodels.ProbabilisticModel._check_output>`
* :py:meth:`ProbabilisticModels.forward_simulate() <abcpy.probabilisticmodels.ProbabilisticModel.forward_simulate>`
* :py:meth:`ProbabilisticModels.get_output_dimension() <abcpy.probabilisticmodels.ProbabilisticModel.get_output_dimension>`

We want our model to work in both described scenarios, so our model also has to conform to the API of
:py:class:`Continuous <abcpy.probabilisticmodels.Continuous>` since the model output, which is the resulting data from a
forward simulation, is from a continuous domain. For completeness, here the abstract methods defined by
:py:class:`Continuous <abcpy.probabilisticmodels.Continuous>` and :py:class:`Discrete
<abcpy.probabilisticmodels.Discrete>`:

* :py:meth:`Continuous.pdf() <abcpy.probabilisticmodels.Continuous.pdf>`
* :py:meth:`Discrete.pmf() <abcpy.probabilisticmodels.Discrete.pmf>`

Initializing a New Model
^^^^^^^^^^^^^^^^^^^^^^^^

Since a Gaussian model generates continous numbers, the newly implement class derives from
:py:class:`Continuous <abcpy.probabilisticmodels.Continuous>` and the header look as follows:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
   :language: python
   :lines: 7

A good way to start implementing a new model is to define a convenient way to initialize it with its input parameters.
In ABCpy all input parameters are either independent ProbabilisticModels or Hyperparameters. Thus, they should not be
stored within but rather referenced in the model we implement. This reference is handled by the
:py:class:`InputConnector <abcpy.probabilisticmodels.InputConnector>` class and **must be used** in our model
implementation. The required procedure is to call the init function of ProbabilisticModels and pass an InputConnector
object to it.

.. automethod::  abcpy.probabilisticmodels.ProbabilisticModel.__init__
    :noindex:

However, it would be very inconvenient to initialize our Gaussian model with an InputConnector object. We rather like
the init function to accept a list of parameters :code:`[mu, sigma]`, where :code:`mu` is the mean and :code:`sigma` is
the standard deviation which are the sole two parameters of our generative Gaussian model. So the idea is to take
a convenient input and transform it it an InputConnection object that in turn can be passed to the initializer of
the super class. This leads to the following implementation:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 12-21
    :dedent: 4
    :linenos:


First, we do some basic syntactic checks on the input that throw exceptions if unreasonable input is provided. Line 9 is
the interesting part: the InputConnector comes with a convenient set of factory methods that create InputConnector
objects:

* :py:meth:`abcpy.probabilisticmodels.InputConnector.from_number()`
* :py:meth:`abcpy.probabilisticmodels.InputConnector.from_model()`
* :py:meth:`abcpy.probabilisticmodels.InputConnector.from_list()`


We use the factory method :py:meth:`from_list <abcpy.probabilisticmodels.InputConnector.from_list>`. The resulting
InputConnector creates links between our Gaussian model and the models (or hyperparameters) that are used for :code:`mu`
and :code:`sigma` at initialization time. For example, if :code:`mu` and :code:`sigma` are initialized as
hyperparameters like

.. code-block:: python

    model = Gaussian([0, 1])

the :code:`from_list()` method will automatically create two HyperParameter objects :code:`HyperParameter(0)` and
:code:`HyperParameter(1)` and will link the our current Gaussian model inputs to them. If we initialize :code:`mu` and
:code:`sigma` with existing models like

.. code-block:: python

    uniform1 = Uniform([-1, 1])
    uniform2 = Uniform([10,20])
    model = Gaussian([uniform1, uniform2])

the :code:`from_list()` method will link our inputs to the uniform models.


Additionally, every model instance should have a unique name, which should also be passed to the init function of the
super class.


Checking the Input
^^^^^^^^^^^^^^^^^^

The next function we implement is :py:meth:`_check_input <abcpy.probabilisticmodels.ProbabilisiticModels._check_input>`
which should behave as described in the documentation:

.. automethod::  abcpy.probabilisticmodels.ProbabilisticModel._check_input
   :noindex:

This leads to the following implementation:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 24-35
    :dedent: 4
    :linenos:

Forward Simulation
^^^^^^^^^^^^^^^^^^

At the core of our model lies the capability to forward simulate and create pseudo observations. To expose this
functionality the following method has to be implemented:

.. automethod::  abcpy.probabilisticmodels.ProbabilisticModel.forward_simulate
   :noindex:

A proper implementation look as follows:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 50-60
    :dedent: 4
    :linenos:

Note that both :code:`mu` and :code:`sigma` are stored in the list input values in the same order as we provided them
to the InputConnector object in the init function. Futher note that the output is a list of vectors, each of dimension
one, though the Gaussian generative model only produces real numbers.


Checking the Output
^^^^^^^^^^^^^^^^^^^

We also need to check the output of the model. This method is commonly used in case our model is used as an input for
other models. When using an inference scheme that utilizes perturbation, the output of our model is slightly perturbed.
We have to make sure that the perturbed output is still valid for our model. The details of implementing the method
:py:meth:`_check_output() <abcpy.probabilisticmodels.ProbabilisiticModels._check_output>` can be found in the
documentation:

.. automethod::  abcpy.probabilisticmodels.ProbabilisticModel._check_output
   :noindex:

Since the output of a Gaussian generative model is a single number from the full real domain, we can restrict ourselves
to syntactic checks. However, one could easily imagine models for which the output it restricted to a certain domain.
Then, this function should return :code:`False` as soon as values are out of the desired domain.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 38-43
    :dedent: 4
    :linenos:

Note that implementing this method is particularly important when using the current model as input for other models,
hence in the second scenario described in `Implementing a new Model`_. In case our model should only be used for the
first scenario, it is safe to omit the check and return true.


Getting the Output Dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have expose the dimension of the produced output of our model using the following method:

.. automethod::  abcpy.probabilisticmodels.ProbabilisticModel.get_output_dimension
   :noindex:

Since our model generates a single float number in one forward simulation, the implementation looks is straight forward:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 46-47
    :dedent: 4
    :linenos:

Note that implementing this method is particularly important when using the current model as input for other models,
hence in the second scenario described in `Implementing a new Model`_. In case our model should only be used for the
first scenario, it is safe to return 1.


Calculating the Probability Density Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since our model also derives from :py:class:`Continuous <abcpy.probabilisticmodels.Continuous>` we also have to implement
the following function that calculates the probability density function at specific point.

.. automethod::  abcpy.probabilisticmodels.Continuous.pdf
   :noindex:

As mentioned above, this is only required if one wants to use our model as input for other models. An implementation looks
as follows:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
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

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_cpp/gaussian_model_simple.cpp
   :language: c++
   :lines: 9 - 17
   :linenos:

To use this code in Python, we need to specify exactly how to expose the C++
function to Python. Therefore, we write a Swig interface file that look as
follows:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_cpp/gaussian_model_simple.i
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

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_cpp/gaussian_model_simple.i
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

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_cpp/pmcabc-gaussian_model_simple.py
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

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_R/gaussian_model.R
    :language: R
    :lines: 1 - 4
    :linenos:

More complex R models are incorporated in the same way. To include this function
within ABCpy we include the following code at the beginning of our Python file:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_R/gaussian_model.py
    :language: python
    :lines: 6 - 14
    :linenos:

This imports the R function :code:`simple_gaussian` into the Python environment. We need to build our own model to
incorporate this R function as in the previous section. The only difference is in the :code:`forward_simulate` method of
the class :code:`Gaussian'.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_R/gaussian_model.py
    :language: python
    :lines: 59
    :dedent: 8
    :linenos:

The default output for R functions in Python is a float vector. This must be
converted into a Python numpy array for the purposes of ABCpy.


Implementing a new Distance
---------------------------
We will now explain how you can implement your own distance measure. A new distance is implemented as a new class that
derives from :py:class`Distance <abcpy.distance.Distance>` and for which the following three methods have to be
implemented:

* :py:meth:`Distance.__init__() <abcpy.distances.Distance.__init__>`
* :py:meth:`Distance.distance() <abcpy.distances.Distance.distance>`
* :py:meth:`Distance.dist_max() <abcpy.distances.Distance.dist_max>`


Let us first look at the initializer documentation:

.. automethod:: abcpy.distances.Distance.__init__
    :noindex:

Distances in ABCpy should act on summary statistics. Therefore, at initialization of a distance calculator, a statistics
calculator should be provided. The following header conforms to this idea:

.. literalinclude:: ../../abcpy/distances.py
    :language: python
    :lines: 112-118
    :dedent: 4

Then, we need to define how the distance is calculated. First we compute the summary statistics from the datasets and
then compute the distance between the summary statistics. Notice, while computing the summary statistics we save the
first dataset and the corresponding summary statistics. This is since we always pass the observed dataset first to the
distance function. The observed dataset does not change during an inference computation and thus it is efficient to
compute it once and store it internally.

.. literalinclude:: ../../abcpy/distances.py
    :language: python
    :lines: 134-146
    :dedent: 4

Finally, we need to define the maximal distance that can be obtained from this distance measure. 

.. literalinclude:: ../../abcpy/distances.py
    :language: python
    :lines: 149-150
    :dedent: 4

The newly defined distance class can be used in the same way as the already existing once. The complete example for this
tutorial can be found `here
<https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/distances/default_distance.py>`_.


Implementing a new Perturbation Kernel
--------------------------------------

To implement a new kernel, we need to implement a new class that derives from
:py:class:`abcpy.perturbationkernel.PerturbationKernel` and that implements the following abstract methods:

* :py:meth:`PerturbationKernel.__init__() <abcpy.perturbationkernel.PerturbationKernel.__init__>`
* :py:meth:`PerturbationKernel.calculate_cov() <abcpy.perturbationkernel.PerturbationKernel.calculate_cov>`
* :py:meth:`PerturbationKernel.update() <abcpy.perturbationkernel.PerturbationKernel.update>`

Kernels in ABCpy can be of two types: they can either be derived from the class :py:class:`ContinuousKernel
<abcpy.perturbationkernel.ContinuousKernel>` or from :py:class:`DiscreteKernel
<abcpy.perturbationkernel.DiscreteKernel>`. In case a continuous kernel is required, the following method must be
implemented:

.. literalinclude:: ../../abcpy/perturbationkernel.py
    :language: python
    :lines: 99

On the other hand, if the kernel is a discrete kernel, we would need the following method:

.. literalinclude:: ../../abcpy/perturbationkernel.py
    :language: python
    :lines: 107

As an example, we will implement a kernel which perturbs continuous parameters using a multivariate normal
distribution (which is already implemented within ABCpy). First, we need to define a constructor.

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.__init__
    :noindex:

Thus, ABCpy expects that the arguments passed to the initializer is of type :py:class:`ProbibilisticModel
<abcpy.probabilisticmodels.ProbabilisticModel>`, which can be seen as the random variables that should be perturbed by
this kernel. All these models should be saved on the kernel for future reference.

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 5, 7,8

Next, we need the following method:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.calculate_cov
    :noindex:

This method calculates the covariance matrix for your kernel. Of course, not all kernels will have covariance matrices.
However, since some kernels do, it is necessary to implement this method for all kernels. *If your kernel does not have
a covariance matrix, simply return an empty list.*

The two arguments passed to this method are the accepted parameters manager and the kernel index. An object of type
:py:class:`AcceptedParameterManager <abcpy.acceptedparametersmanager.AcceptedParametersManager>` is always initialized
when an inference method object is instantiated. On this object, the accepted parameters, accepted weights, accepted
covariance matrices for all kernels and other information is stored. This is such that various objects can access this
information without much hassle centrally. To access any of the quantities mentioned above, you will have to call the
`.value()` method of the corresponding quantity.

The second parameter, the kernel index, specifies the index of the kernel in the list of kernels that the inference
method will in the end obtain. Since the user is expected to collect all his kernels in one object, this index will
automatically be provided. You do not need any knowledge of what the index actually is. However, it is used to access
the values relevant to your kernel, for example the current calculated covariance matrix for a kernel.

Let us now look at the implementation of the method:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 10, 25-30
    :dedent: 4

Some of the implemented inference algorithms weigh different sets of parameters differently. Therefore, if such weights
are provided, we would like to weight the covariance matrix accordingly. We, therefore, check whether the accepted
parameters manager contains any weights. If it does, we retrieve these weights, and calculate the covariance matrix
using numpy, the parameters relevant to this kernel and the weights. If there are no weights, we simply calculate an
unweighted covariance matrix.

Next, we need the method:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.update
    :noindex:

This method perturbs the parameters that are associated with the random variables the kernel should perturb. The method
again requires an accepted parameters manager and a kernel index. These have the same meaning as in the last method. In
addition to this, a row index is required, as well as a random number generator. The row index specifies which set of
parameters should be perturbed. There are usually multiple sets, which should be perturbed by different workers during
parallelization. We, again, need not to worry about the actual value of this index.

The random number generator should be a random number generator compatible with numpy. This is due to the fact that
other methods will pass their random number generator to this method, and all random number generators used within ABCpy
are provided by numpy. Also, note that even if your kernel does not require a random number generator, you still need to
pass this argument.

Here the implementation for our kernel:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 32, 53, 56-60
    :dedent: 4

The first line shows how you obtain the values of the parameters that your kernel should perturb. These values are
converted to a numpy array. Then, the covariance matrix is retrieved from the accepted parameters manager using a
similar function call. Finally, the parameters are perturbed and returned.

Last but not least, each kernel requires a probability density or probability mass function depending on whether it is a
Continous Kernel or a Discrete Kernel:

.. automethod:: abcpy.perturbationkernel.PerturbationKernel.pdf
    :noindex:

This method is implemented as follows for the multivariate normal:

.. literalinclude:: ../../examples/extensions/perturbationkernels/multivariate_normal_kernel.py
    :language: python
    :lines: 62, 83-87
    :dedent: 4

We simply obtain the parameter values and covariance matrix for this kernel and calculate the probability density
function using SciPy.

Note that after defining your own kernel, you will need to collect all your kernels in a
:py:class:`JointPerturbationKernel <abcpy.perturbationkernel.JointPerturbationKernel>` object in order for inference to
work. For an example on how to do this, check the :ref:`Using perturbation kernels <gettingstarted>` section.

The complete example used in this tutorial can be found `here
<https://github.com/eth-cscs/abcpy/blob/master/examples/extensions/perturbationkernels/multivariate_normal_kernel.py>`_.
