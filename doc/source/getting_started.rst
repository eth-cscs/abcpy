.. _gettingstarted:

2. Getting Started
==================

Here, we explain how to use ABCpy to quntify parameter uncertainty of a probabbilistic model given some observed
dataset. If you are new to uncertainty quantification using Approximate Bayesian Computation (ABC), we recommend you to
start with the `Parameters as Random Variables`_ section.

Parameters as Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an example, if we have measurements of the height of a group of grown up human and it is also known that a Gaussian
distribution is an appropriate probabilistic model for these kind of observations, then our observed dataset would be
measurement of heights and the probabilistic model would be Gaussian.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 72
    :dedent: 4

The Gaussian or Normal model has two parameters: the mean, denoted by :math:`\mu`, and the standard deviation, denoted
by :math:`\sigma`. We consider these parameters as random variables. The goal of ABC is to quantify the uncertainty of
these parameters from the information contained in the observed data.

In ABCpy, a :py:class:`abcpy.probabilisticmodels.ProbabilisticModel` object represents probabilistic relationship
between random variables or between random variables and observed data. Each of the :py:class:`ProbabilisticModel
<abcpy.probabilisticmodels.ProbabilisticModel>` object has a number of input parameters: they are either random
variables (output of another :py:class:`ProbabilisticModel <abcpy.probabilisticmodels.ProbabilisticModel>` object) or
constant values and considered known to the user (:py:class:`Hyperparameters
<abcpy.probabilisticmodels.Hyperparameters>`). If you are interested in implementing your own probabilistic model,
please check the implementing a new model section.

To define a parameter of a model as a random variable, you start by assigning a *prior* distribution on them. We can
utilize *prior* knowledge about these parameters as *prior* distribution. In the absence of prior knowledge, we still
need to provide *prior* information and a non-informative flat disribution on the parameter space can be used. The prior
distribution on the random variables are assigned by a probabilistic model which can take other random variables or hyper
parameters as input.

In our Gaussian example, providing prior information is quite simple. We know from experience that the average height
should be somewhere between 150cm and 200cm, while the standard deviation is around 5 to 25. In code, this would look as
follows:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 74-76, 78-79
    :dedent: 4
    :linenos:

We have defined the parameter :math:`\mu` and :math:`\sigma` of the Gaussian model as random variables and assigned
Uniform prior distributions on them. The parameters of the prior distribution :math:`(150, 200, 5, 25)` are assumed to
be known to the user, hence they are called hyperparameters. Also, internally, the hyperparameters are converted to
:py:class:`Hyperparameter <abcpy.probabilisticmodels.HyperParameters>` objects. Note that you are also required to pass
a *name* string while defining a random variable. In the final output, you will see these names, together with the
relevant outputs corresponding to them.

For uncertainty quantification, we follow the philosophy of Bayesian inference. We are interested in the distribution of
the parameters we get after incorporating the information that is implicit in the observed dataset with the prior
information. This target distribution is called *posterior* distribution of the parameters. For inference, we draw
independent and identical sample values from the posterior distribution. These sampled values are called posterior
samples and used to either approximate the posterior distribution or to integrate in a Monte Carlo style w.r.t the
posterior distribution. The posterior samples are the ones you get as a result of applying an ABC inference scheme.

The heart of the ABC inferential algorithm is a measure of discrepancy between the observed dataset and the synthetic
dataset (simulated/generated from the model). Often, computation of discrepancy measure between the observed and
synthetic dataset is not feasible (e.g., high dimensionality of dataset, computationally to complex) and the discrepancy
measure is defined by computing a distance between relevant *summary statistics* extracted from the datasets. Here we
first define a way to extract *summary statistics* from the dataset.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 82-83
    :dedent: 4
    :linenos:

Next we define the discrepancy measure between the datasets, by defining a distance function (LogReg distance is chosen
here) between the extracted summary statistics. If we want to define the discrepancy measure through a distance function
between the datasets directly, we choose Identity as summary statistics which gives the original dataset as the
extracted summary statistics. This essentially means that the distance object automatically extracts the statistics from
the datasets, and then compute the distance between the two statistics.
 
.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 86-87
    :dedent: 4

Algorithms in ABCpy often require a perturbation kernel, a tool to explore the parameter space. Here, we use the default
kernel provided, which explores the parameter space of random variables, by using e.g. a multivariate Gaussian
distribution or by performing a random walk depending on whether the  corresponding random variable is continuous or
discrete. For a more involved example, please consult `Complex Perturbation Kernels`_.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 90-91
    :dedent: 4

Finally, we need to specify a backend that determins the parallelization framework to use. The example code here uses
the dummy backend :py:class:`BackendDummy <abcpy.backends.dummy>` which does not parallelize the computation of the
inference schemes, but  which this is handy for prototyping and testing. For more advanced parallelization backends
available in ABCpy, please consult :ref:`Using Parallelization Backends <parallelization>` section.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 95-96
    :dedent: 4

In this example, we choose PMCABC algorithm (inference scheme) to draw posterior samples of the parameters. Therefore,
we instantiate a PMCABC object by passing the random variable corresponding to the observed dataset, the distance
function, backend object, perturbation kernel and a seed for the random number generator.

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 98-100
    :dedent: 4

Finally, we can parametrize the sampler and start sampling from the posterior distribution of the parameters given the
observed dataset:

.. literalinclude:: ../../examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py
    :language: python
    :lines: 102-106
    :dedent: 4

The above inference scheme gives us samples from the posterior distribution of the parameter of interest *height*
quantifying the uncertainty of the inferred parameter, which are stored in the journal object. See :ref:`Post Analysis
<postanalysis>` for further information on extracting results.

Note that the model and the observations are given as a list. This is due to the fact that in ABCpy, it is possible to
have hierarchical models, building relationships between co-occuring groups of datasets. To learn more, see the
`Hierarchical Model`_ section.

The full source can be found in `examples/extensions/simplemodels/gaussian_python/pmcabc_gaussian_model_simple.py`. To
execute the code you only need to run

::

   python3 pmcabc_gaussian_model_simple.py


Probabilistic Dependency between Random Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since release 0.5.0 of ABCpy, a probabilistic dependency structures (e.g., a Bayesian network) between random variables
can be modelled. Behind the scene, ABCpy will represent this dependency structure as a directed acyclic graph (DAG) such
that the inference can be done on the full graph. Further we can also define new random variables through operations
between existing random variables. To make this concept more approachable, we now exemplify an inference problem on a
probabilistic dependency structure.

Let us consider the following example: Assume we have a school with different classes. Each class has some number of
students. All students of the school take an exam and receive some grade. Lets consider the grades of the students as
our observed dataset:

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 6
    :dedent: 4

The grades of the students depend on several variables: if there were bias, the size of the class a student is in, as
well as the social background of the student. The dependency structure between these variables can be defined using the
following Bayesian network:

.. image:: network_1_model.png

Here we assume the size of a class the student belongs to and his/her social background are normally distributed with
some mean and variance. However, they also both depend on the location of the school. In certain neighbourhoods, the
class size might be larger, or the social background might differ. We therefore have additional parameters, specifying
the mean of these normal distributions will vary uniformly between some lower and upper bounds. Finally, we can assume
that the grade without any bias would be a normally distributed parameter around an average grade.

We can define these random variables and the dependencies between them (in a nutshell, a Bayesian network) in ABCpy in
the following way:

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 9-10, 13, 16, 19
    :dedent: 4

So, each student will receive some grade without additional effects which is normally distributed, but then the final
grade recieved will be a function of grade without additional effects and the other random variables defined beforehand
(e.g., `school_location`, `class_size` and `background`). The model for the final grade of the students now can be
written as:

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 22
    :dedent: 4

Notice here we created  a new random variable `final_grade`, by subtracting the random variables `class_size` and
`background` from the random variable `grade`. In short, this illustrates that you can perform standard operations "+",
"-", "*", "/" and "**" (the power operator in Python) on any two random variables, to get a new random variable. It is
possible to perform these operations between two random variables additionally to the general data types of Python
(integer, float, and so on) since they are converted to :py:class:`HyperParameters
<abcpy.probabilisticmodels.HyperParameters>`.

Please keep in mind that **parameters defined via operations will not be included in your list of parameters in the
journal file**. However, all parameters that are part of the operation, and are not fixed, will be included, so you can
easily perform the required operations on the final result to get these parameters, if necessary. In addition, you can
now also use the :code:`[]` operator (the access operator in Python). This allows you to select single values or ranges
of a multidimensional random variable as a parameter of a new random variable.

To do inference on the Baysian network given the observed datat, we again have to define some summary statistics of our
data (similar to the `Parameters as Random Variables`_ section).

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 25-26
    :dedent: 4

And just as before, we need to provide a distance measure, a backend and a kernel to the inference scheme of our choice,
which again is PMCABC.

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 29-30, 33-34, 37-38
    :dedent: 4

Finally, we can parametrize the sampler and start sampling from the posterior distribution:

.. literalinclude:: ../../examples/extensions/graphicalmodels/pmcabc_graphical_model.py
    :language: python
    :lines: 41-43, 46-47, 50
    :dedent: 4

The source code can be found in `examples/extensions/graphicalmodels/pmcabc_graphical_model.py` and can be run as show
above.

Hierarchical Model
~~~~~~~~~~~~~~~~~~

As mentioned above, ABCpy supports inference on hierarchical models. Hierarchical models can share some of the
parameters of interest and model co-occuring datasets. To illustrate how this is implemented, we will follow the example
model given in `Probabilistic Dependency between Random Variables`_ section and extend it to co-occuring datasets.

.. image:: network.png

Let us assume that in addition to the final grade of a student, we have (co-occuring, observed) data for final
scholarships given out by the school to the students. Whether a student gets a scholarship depends on his social
background and on an independent score.

.. literalinclude:: ../../examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py
    :language: python
    :lines: 25
    :dedent: 4

We assume the score is normally distributed and we model it as follows:

.. literalinclude:: ../../examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py
    :language: python
    :lines: 28
    :dedent: 4

We model the impact of the students social background on the scholarship as follows:

.. literalinclude:: ../../examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py
    :language: python
    :lines: 31
    :dedent: 4

With this we now have two *root* ProbabilisicModels (random variables), namely :code:`final_grade` and
:code:`final_scholarship`, whose output can directly compared to the observed datasets :code:`grade_obs` and
:code:`scholarship_obs`. With this we are able to do an inference computation on all free parameters of the hierarchical
model (of the DAG) given our observations.

To inference uncertainty of our parameters, we follow the same steps as in our previous examples: We choose summary
statistics, distance, inference scheme, backend and kernel. We will skip the definitions that have not changed from the
previous section. However, we would like to point out the difference in definition of the distance. Since we are now
considering two observed datasets, we need to define an distances on them separately. Here, we use the Euclidean
distance for each observed data set and corresponding simulated dataset. You can use two different distances on two
different observed datasets.

.. literalinclude:: ../../examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py
    :language: python
    :lines: 33-41
    :dedent: 4

Using these two distance functions with the final code look as follows:

.. literalinclude:: ../../examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py
    :language: python
    :lines: 43-64
    :dedent: 4

Observe that the lists given to the sampler and the sampling method now contain two entries. These correspond to the two
different observed data sets respectively. Also notice now we provide two different distances corresponding to the two
different root models and their observed datasets. Presently ABCpy combines the distances by a linear combination, however
customized combination strategies can be implemented by the user.

The full source code can be found in `examples/extensions/hierarchicalmodels/pmcabc_hierarchical_models.py`.


Complex Perturbation Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As pointed out earlier, it is possible to define complex perturbation kernels, perturbing different random variables in
different ways. Let us take the same example as in the `Hierarchical Model`_ and assume that we want to perturb the
schools location as well as the scholarship score, using a multivariate normal kernel. However, the remaining
parameters we would like to perturb using a multivariate Student's-T kernel. This can be implemented as follows:

.. literalinclude:: ../../examples/extensions/perturbationkernels/pmcabc_perturbation_kernels.py
    :language: python
    :lines: 44-46
    :dedent: 4

We have now defined how each set of parameters is perturbed on its own. The sampler object, however, needs to be
provided with one single kernel. We, therefore, provide a class which groups the above kernels together. This class,
:py:class:`abcpy.perturbationkernel.JointPerturbationKernel`, knows how to perturb each set of parameters individually.
It just needs to be provided with all the relevant kernels:

.. literalinclude:: ../../examples/extensions/perturbationkernels/pmcabc_perturbation_kernels.py
    :language: python
    :lines: 49-50
    :dedent: 4

This is all that needs to be changed. The rest of the implementation works the exact same as in the previous example. If
you would like to implement your own perturbation kernel, please check :ref:`Implementing a new Perturbation Kernel
<implementations>`. Please keep in mind that you can only perturb parameters. **You cannot use the access operator to
perturb one component of a multi-dimensional random variable differently than another component of the same variable.**

The source code to this section can be found in `examples/extensions/perturbationkernels/pmcabc_perturbation_kernels.py`

Inference Schemes
~~~~~~~~~~~~~~~~~

In ABCpy, we implement widely used and advanced variants of ABC inferential schemes:

* Rejection ABC :py:class:`abcpy.inferences.RejectionABC`,
* Population Monte Carlo ABC :py:class:`abcpy.inferences.PMCABC`,
* Sequential Monte Carlo ABC :py:class:`abcpy.inferences.SMCABC`,
* Replenishment sequential Monte Carlo ABC (RSMC-ABC) :py:class:`abcpy.inferences.RSMCABC`,
* Adaptive population Monte Carlo ABC (APMC-ABC) :py:class:`abcpy.inferences.APMCABC`,
* ABC with subset simulation (ABCsubsim) :py:class:`abcpy.inferences.ABCsubsim`, and
* Simulated annealing ABC (SABC) :py:class:`abcpy.inferences.SABC`.

To perform ABC algorithms, we provide different standard distance functions between datasets, e.g., a discrepancy
measured by achievable classification accuracy between two datasets

* :py:class:`abcpy.distances.Euclidean`,
* :py:class:`abcpy.distances.LogReg`,
* :py:class:`abcpy.distances.PenLogReg`.

We also have implemented the population Monte Carlo :py:class:`abcpy.inferences.PMC` algorithm to infer parameters when
the likelihood or approximate likelihood function is available. For approximation of the likelihood function we provide
two methods:

* Synthetic likelihood approximation :py:class:`abcpy.approx_lhd.SynLiklihood`, and another method using
* penalized logistic regression :py:class:`abcpy.approx_lhd.PenLogReg`.

Next we explain how we can use PMC algorithm using approximation of the likelihood functions. As we are now considering
two observed datasets corresponding to two root models, we need to define an approximation of likelihood function for
each of them separately. Here, we use the :py:class:`abcpy.approx_lhd.SynLiklihood` for each of the root models. It is
also possible to use two different approximate likelihoods for two different root models.

.. literalinclude:: ../../examples/extensions/approx_lhd/pmc_hierarchical_models.py
    :language: python
    :lines: 33-41
    :dedent: 4

We then parametrize the sampler and sample from the posterior distribution. 

.. literalinclude:: ../../examples/extensions/approx_lhd/pmc_hierarchical_models.py
    :language: python
    :lines: 52-64
    :dedent: 4

Observe that the lists given to the sampler and the sampling method now contain two entries. These correspond to the two
different observed data sets respectively. Also notice now we provide two different distances corresponding to the two
different root models and their observed datasets. Presently ABCpy combines the distances by a linear combination.
Further possibilities of combination will be made available in later versions of ABCpy.

The source code can be found in `examples/extensions/approx_lhd/pmc_hierarchical_models.py`.


Summary Selection
~~~~~~~~~~~~~~~~~

We have noticed in the `Parameters as Random Variables`_ Section, the discrepancy measure between two datasets is
defined by a distance function between extracted summary statistics from the datasets. Hence, the ABC algorithms are
subjective to the summary statistics choice. This subjectivity can be avoided by a data-driven summary statistics choice
from the available summary statistics of the dataset. In ABCpy we provide a semi-automatic summary selection procedure in
:py:class:`abcpy.summaryselections.Semiautomatic`

Taking our initial example from `Parameters as Random Variables`_ where we model the height of humans, we can had summary
statistics defined as follows:

.. literalinclude:: ../../examples/extensions/summaryselection/pmcabc_gaussian_summary_selection.py
    :language: python
    :lines: 21-23
    :dedent: 4

Then we can learn the optimized summary statistics from the given list of summary statistics using the semi-automatic
summary selection procedure as following:

.. literalinclude:: ../../examples/extensions/summaryselection/pmcabc_gaussian_summary_selection.py
    :language: python
    :lines: 25-32
    :dedent: 4

Then we can perform the inference as before, but the distances will be computed on the newly learned summary statistics
using the semi-automatic summary selection procedure.

Model Selection
~~~~~~~~~~~~~~~

A further extension of the inferential problem is the selection of a model (M), given an observed
dataset, from a set of possible models. The package also includes a parallelized version of random 
forest ensemble model selection algorithm [:py:class:`abcpy.modelselections.RandomForest`].

Lets consider an array of two models `Normal` and `StudentT`. We want to find out which one of these two models are the
most suitable one for the observed dataset `y_obs`.

.. literalinclude:: ../../examples/extensions/modelselection/randomforest_modelselections.py
    :language: python
    :lines: 7-19
    :dedent: 4

We first need to initiate the Model Selection scheme, for which we need to define the summary statistics and backend:

.. literalinclude:: ../../examples/extensions/modelselection/randomforest_modelselections.py
    :language: python
    :lines: 21-30
    :dedent: 4

Now we can choose the most suitable model for the observed dataset `y_obs`,

.. literalinclude:: ../../examples/extensions/modelselection/randomforest_modelselections.py
    :language: python
    :lines: 32-33
    :dedent: 4

or compute posterior probability of each of the models given the observed dataset.

.. literalinclude:: ../../examples/extensions/modelselection/randomforest_modelselections.py
    :language: python
    :lines: 35-36
    :dedent: 4







