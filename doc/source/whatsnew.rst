.. _whatsnew:

2. What's new?
==============

It is now possible to implement bayesian networks in ABCpy. If you are new to bayesian networks, check the section :ref:`Bayesian networks - an introduction <bayes_nets>`.

In this section, we will introduce the various new features for this version of ABCpy.

Random variables and hierarchical models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In short, the following is now possible:

::

    parameter = Normal([[1],[0.1]])
    model = Normal([parametere, [0.2]])

Check out :ref:`Using random variables <gettingstarted>` for an example of how to implement things discussed in this section.

In ABCpy, a :py:class:`abcpy.probabilisticmodels.ProbabilisticModel` object represents a random variable. This includes random variables that correspond to observed data, models. The part of the graph that isn't a model is usually called the *prior*.

Since there are many properties associated with these objects, we did not simply call them random variable or distribution. For you to perform inference on some data, it is not necessary to understand these properties. If you are interested in implementing your own random variable, please check the :ref:`Implementing a new Model <implementations>` section  below.

The way a bayesian network is now built within ABCpy is the following: each :py:class:`abcpy.probabilisticmodels.ProbabilisticModel` object has some input parameters. Each of these parameters can either be a 'fixed' and known to the user, for bayesian networks often called hyperparameter, or another random variable. Behind the scene, ABCpy will ensure that the graph structure is implemented and that inference will be performed on the whole construct.


Random variables do usually not have a fixed value. However, to perform inference, we of course need to sample values from the random variables and use those to simulate data. These sampled values we will usually call parameter values. These values are what you will get as an end result of the inference.

An important thing to keep in mind: **random variables are initialized without sampling values from them immediately.** This works for inference algorithms, since these algorithms will contain a statement that samples from the prior first. However, if you want to implement a graph and then sample from a random variable within the graph structure, the necessary parameter values will not yet be defined. Therefore, if you want to have a 'stand-alone' distribution or model, you will need to call the method `sample_parameters()` for any parameter in the graph, which is best done right after initializing the relevant parameters. Only after this step is performed can you call the `sample_from_distribution()` method to obtain samples from the distribution.

Some words on performance: all operations on this graph structure are performed using recursion. This will, in general, slow down your code compared to previous versions of ABCpy, even if the prior information you give is the exact same as before. However, this will not affect the time inference needs to complete for most use cases, where you define your own hierarchical model. This is due to the fact that simulation is usually a lot slower than traversing the graph. Even if you do not specify such a complicated model, your run time should still be acceptable, given that you do not implement large graphs. Due to the limitations of ABC regarding low dimensionality of your problem, this should, however, not be an issue.


Multiple hierarchical models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out :ref:`Using multiple hierarchical models <gettingstarted>` for an example on how to implement things discussed in this section.

It is now also possible to define multiple models on which the inference acts. The graphs given for each hierarchical model can be connected to those of different models or they can be independent (however, it is not possible for a model to be input to another model within the same round of inference). Note though that of course, for each model the graph has to consist of only one weakly connected component. Otherwise, there is no way for the algorithm to know how to connect the information on different nodes to affect the model.

Operations
~~~~~~~~~~

In short, the following is now possible:

::

    parameter = MultivariateNormal([[1,2],[[0.1,0],[0,0.1]])
    model = 2*parameter[1]+parameter[0]

Check out :ref:`Using operations <gettingstarted>` for an example on how to implement things discussed in this section.

Not only can you now connect random variables through their input parameters, it is also possible to perform operations on these objects to produce new random variables. In short, this means that you can now perform the operations "+", "-", "*", "/" and "**" (the power operator in Python) on any two nodes of your graph, giving a new node. It is possible to perform these operations between two random variables or between random variables and general data types of Python (integer, float, and so on).

Please keep in mind that **parameters defined via operations will not be included in your list of parameters in the journal file**. However, all parameters that are part of the operation, and are not fixed, will be included, so you can easily perform the required operations on the final result to get these parameters, if necessary.

In addition to these operators, you can now also use the "[]" operator (the access operator in Python). This allows you to only use selected values from a multidimensional random variable sample as a parameter of a new random variable.

Perturbation kernels
~~~~~~~~~~~~~~~~~~~~

In short, this is now how you define kernels on different parameters:

::

    kernel = MultivariateNormalKernel([parameter_1, parameter_2])
    kernel_joint = JointPerturbationKernel([kernel])

Check out :ref:`Using perturbation kernels <gettingstarted>` for an example on how to implement things discussed in this section.

Since you can now define much more complicated priors, the kernels have changed accordingly. There are two classes, :py:class:`abcpy.perturbationkernel.ContinuousKernel` and :py:class:`abcpy.perturbationkernel.DiscreteKernel`. The only difference between these two is that the first class acts on continuous random variables (and, therefore, has a probability density function), while the second class acts on discrete random variables (and, therefore, has a probability mass function).

For each kernel you define, you can specify which random variables of the graph it should perturb. In the end, you join all kernels using an object of type :py:class:`abcpy.perturbationkernel.JointPerturbationKernel`. Note that you need to join your kernels, even if you only defined one kernel! This is due to the fact that the algorithm cannot know how many kernels you defined otherwise.

Each algorithm is provided with a default kernel. If you do not specify another kernel and pass it to the sampling object, it will automatically create its own kernel. It will perturb all continuous parameters using a multivariate normal and all discrete parameters using a random walk.

Please keep in mind that you can only perturb whole parameters. **You cannot use the access operator to perturb one part of a random variable differently than another part of the same variable.**