.. _gettingstarted:

3. Getting Started
==================

Here, we explain how to use ABCpy to quntify uncertainty of parameters of a model for some observed dataset.

If you are new to uncertainty quantification using Approximate Bayesian Computation (ABC), we recommend you to start with the `Random variables`_ section. If you would like to see all features in action together, check out `Complex perturbation kernels`_.

Random variables
~~~~~~~~~~~~~~~~~~~~~~

As an example, if we have measurements of the height of a group of grown up human and it is also known that a Gaussian distribution is appropriate to model these kind of observations, then our observed dataset would be measurement of heights and the model would be Gaussian.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 5
    :dedent: 4

Gaussian or Normal model has two parameters: the mean, denoted by :math:`\mu`, and the standard deviation, denoted by :math:`\sigma`. We consider these parameters as random variables. The goal of ABC is to quantify uncertainty of these parameters from the information contained in the observed data.

In ABC, we can utilize *prior* knowledge about these random variables. In this example, it is quite simple. We know from experience that the average height should be somewhere between 150cm and 200cm, while the standard deviation is around 5 to 25. 

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 8-10, 12-13
    :dedent: 4

While that we also provide *name* of the parameters to the prior. You are required to pass such a string for any parameters that is part of the prior. In the final output, you will see these names, together with the values that were infered for this parameter. (In the absence of prior knowledge, we still need to provide *prior* and which should be a non-informative prior.)

For ABC, the inference is done by a measure of discrepancy between the observed dataset and the synthetic dataset (simulated from the model). Often, computation of discrepancy measure between the observed and synthetic dataset is not feasible (e.g., high dimensionality of dataset) and the discrepancy measure is defined by computing a distance between relevant *summary statistics* extracted from the datasets. Here we first define a way to extract *summary statistics* from the dataset.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 16-17
    :dedent: 4

Next we define the discrepancy measure between the datasets, by defining a distance function (LogReg distance is chosen here) between the extracted summary statistics. If we want to define the discrepancy measure through a distance function between the datasets directly, we choose Identity as summary statistics which gives the original dataset as the extracted summary statistics. 
 
.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 20-21
    :dedent: 4

Let us choose PMCABC algorithm for inference of the parameters here. 

Algorithms in ABCpy requires a perturbation kernel, a tool to explore the parameter space. Here, we use the default kernel provided, which explores the parameter space of random variables, by perturbing it using a multivariate Gaussian distribution or performing a random walk if the  corresponding random variable is continuous or discrete. For a more involved example, please consult `Complex perturbation kernels`_.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 24-25
    :dedent: 4

We need to define a backend declaring the type of parallelization. The example code here uses the dummy backend `BackendDummy` which does not parallelize the execution of the inference schemes. But this is handy for prototyping and testing.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 29-30
    :dedent: 4

Now we are ready to instantiate a PMCABC object and pass the kernel and backend objects to the constructor:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 33-34
    :dedent: 4

Finally, we can parametrize the sampler and start sampling of the highly probable values of the parameters (proportional to their probability of occurance) given the observed dataset:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 37-40
    :dedent: 4

The above inference scheme gives us samples from the distribution of the parameter of interest quantifying the uncertainty of the inferred parameter, which are stored in the journal object. See :ref:`Post Analysis <postanalysis>` for further information on extracting results.

Note that the model and the observations are given as a list. This is due to the fact that in ABCpy, it is possible to have Multi-view models, building relationships between co-occuring groups of datasets. To learn more, see the `Multi-view models`_ section.

The full source can be found in `examples/backends/dummy/pmcabc_gaussian.py`.

To execute the code you only need to run

::

   python3 pmcabc_gaussian.py


Multi-level Models
~~~~~~~~~~~~~~~~

Since release XX.XX of ABCpy, we can have complex dependency structures (e.g., a Bayesian network) between random variables. Further we can also define new random variables through operations between existing random variables. In the following we describe an inference problem illustrating both of these ideas. 

For this tutorial, we consider a school with different classes. Each class has some number of students. All students of the school take an exam and receive some grade. Lets consider the grades of the students as our observed dataset:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 6
    :dedent: 4

The grades of the students depend on several variables: if there were bias, the size of the class a student is in, as well as the social background of the student. The dependency structure between these variables can be defined using the following Bayesian network: 

.. image:: network_1_model.png


Here we assume the size of a class the student belongs to and his/her social background are normally distributed with some mean and variance. However, they also both depend on the location of the school. In certain neighbourhoods, the class size might be larger, or the social background might differ. 

We, therefore, have additional parameters, specifying the mean of these normal distributions will vary uniformly between some lower and upper bounds. Finally, we can assume that the grade without any bias would be a normally distributed parameter around an average grade.

We can define these random variables and the dependencies between them (in a nutshell, a Bayesian network) in ABCpy:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 9-10, 13, 16, 19
    :dedent: 4

So, each student will receive some grade which is normally distributed, but this grade is determined by the other random variables defined. The model for the grade of the students now can be written as:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 22
    :dedent: 4

As in the `Random variables`_ section, we now need to define a summary statistics to extract relevant properties of our data.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 25-26
    :dedent: 4

And just as before, we need to provide a distance measure, a backend and a kernel.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 29-30, 33-34, 37-38
    :dedent: 4

Now, again, we can parametrize the sampler and start sampling:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_operations.py
    :language: python
    :lines: 41-43, 46-47, 50
    :dedent: 4

The source code can be found in `examples/backends/dummy/pmcabc_operations.py`.


Multi-view models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, ABCpy supports inference on multiple hierarchical models at once. These hierarchical models can share some of the parameters of interest. To illustrate how this is implemented, we will extend the model given in `Multi-level Models`_.

.. image:: network.png

As in the last section, our first model consists of the final grades students receive in an exam. This final grade depends on an average grade, the size of the class of each student and the students social background. The latter two variables also depend on the location of the school.

For an explanation of the code, please see `Multi-level Models`_.

We now consider the second hierarchical model. First, we again need to define our observed data set:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_multiple_models.py
    :language: python
    :lines: 25
    :dedent: 4

Let us assume that the school gives out scholarships to students. Each student receives some score, which determines whether he will receive a scholarship or not. We assume that this score is normally distributed:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_multiple_models.py
    :language: python
    :lines: 28
    :dedent: 4

However, depending on the students social background, the score will be changed, giving us the definition of our second hierarchical model.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_multiple_models.py
    :language: python
    :lines: 31
    :dedent: 4

As in the algorithms before, we now define summary statistics, distance, backend and kernel. We will skip the definitions that have not changed from the previous section. However, we would like to point out the difference in definition of the distance.

Since we are now considering two hierarchical models, we need to define an overall distance on the two. Here, we use the default distance provided in ABCpy. It uses the euclidean distance for each hierarchical model and corresponding data set seperatly. All distances are added, and the result is divided by the number of data sets given. If you would like to implement a different distance measure on multiple data sets, check :ref:`Implementing a new Distance <implementations>`.

.. literalinclude:: ../../examples/backends/dummy/pmcabc_multiple_models.py
    :language: python
    :lines: 38-39
    :dedent: 4


We then parametrize the sampler and sample:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_multiple_models.py
    :language: python
    :lines: 50-52, 55-56, 59
    :dedent: 4

Observe that the lists given to the sampler and the sampling method now contain two entries. These correspond to the two hierarchical models and the observed data for the two models, respectively.

The source code can be found in `examples/backends/dummy/pmcabc_multiple_models.py`.


Complex perturbation kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

We will consider the same example as in the `Multi-view models`_ section.

As pointed out earlier, it is possible to define complex perturbation kernels, perturbing random variables in different ways. Let us assume that we want to perturb the schools location, as well as the scholarship score, using a multivariate normal kernel. However, the remaining parameters we would like to perturb using a multivariate Student's-T kernel.

This can be implemented as following:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_perturbation_kernels.py
    :language: python
    :lines: 44-46
    :dedent: 4

We have now defined how each set of parameters is perturbed on its own. The sampler object, however, needs to be provided with one single kernel. We, therefore, provide a class which groups the above kernels together. This class, :py:class:`abcpy.perturbationkernel.JointPerturbationKernel`, knows how to perturb each set of parameters individually. It just needs to be provided with all the relevant kernels:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_perturbation_kernels.py
    :language: python
    :lines: 49-50
    :dedent: 4

This is all that needs to be changed. The rest of the implementation works the exact same as in the previous example. If you would like to implement your own perturbation kernel, please check :ref:`Implementing a new Perturbation Kernel <implementations>`.

The source code to this section can be found in `examples/backends/dummy/pmcabc_perturbation_kernels.py`
