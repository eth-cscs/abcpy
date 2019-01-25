.. _parallelization:

4. Parallelization Backends
===========================

Using Parallelization Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running ABC algorithms is often computationally expensive, thus ABCpy is built
with parallelization in mind. In order to run your inference schemes in parallel
on multiple nodes (computers) you can choose from the following backends.

Using the MPI Backend
~~~~~~~~~~~~~~~~~~~~~

To run ABCpy in parallel using MPI, one only needs to use the provided MPI
backend. Using the same example as before, the statements for the backend have to
be changed to

.. literalinclude:: ../../examples/backends/mpi/pmcabc_gaussian.py
    :language: python
    :lines: 6-10
    :dedent: 4

In words, one only needs to initialize an instance of the MPI backend. The
number of ranks to spawn are specified at runtime through the way the script is
run. A minimum of two ranks is required, since rank 0 (master) is used to
orchestrate the calculation and all other ranks (workers) actually perform the
calculation. (The default value of `process_per_model` is 1. If your simulator 
model is not parallelized using MPI, do not specify
`process_per_model > 1`. The use of `process_per_model` for nested parallelization 
will be explained below.)

The standard way to run the script using MPI is directly via mpirun like below
or on a cluster through a job scheduler like Slurm:

::

   mpirun -np 4 python3 pmcabc_gaussian.py


The adapted Python code can be found in
`examples/backend/mpi/pmcabc_gaussian.py`.

Nested-MPI parallelization for MPI-parallelized simulator models
------------------------------------------------------------------

Sometimes, the simulator model itself has 
large compute requirements and needs parallelization. To achieve this parallelization
using threads, the MPI backend need to be configured such that each MPI
rank can spawn multiple threads on a node. However, there might be situations
where node-local parallelization using threads is not sufficient and
parallelization across nodes is required.

Parallelization of the forward model across nodes is possible *but limited* to
the MPI backend. Technically, this is implemented using individual MPI
communicators for each forward model. The number of ranks per communicator 
(defined as: `process_per_model`)
can be passed at the initialization of the backend as follows:

.. literalinclude:: ../../examples/backends/mpi/mpi_model_inferences.py
    :language: python
    :lines: 10-11
    :dedent: 4

Here each model is assigned a MPI communicator with 2 ranks. Clearly, the MPI
job has to be configured manually such that the total amount of MPI ranks is ideally
a multiple of the ranks per communicator plus one additional rank for the
master. For example, if we want to run n instances of a MPI model and allows m
processes to each instance, we will have to spawn (n*m)+1 ranks.

For `forward_simulation` of the MPI-parallelized simulator model has to be able 
to take an MPI communicator as a parameter.

An example of an MPI-parallelized simulator model, which can be used with ABCpy 
nested-parallelization, can be found in `examples/backend/mpi/mpi_model_inferences.py`.
The `forward_simulation` function of the above model is as follows:

.. literalinclude:: ../../examples/backends/mpi/mpi_model_inferences.py
    :language: python
    :lines: 48-77
    :dedent: 4

Note that in order to run jobs in parallel you need to have MPI installed on the
system(s) in question with the requisite Python bindings for MPI (mpi4py). The
dependencies of the MPI backend can be install with
`pip install -r requirements/backend-mpi.txt`.

Details on the installation can be found on the official `Open MPI homepage
<https://www.open-mpi.org/>`_ and the `mpi4py homepage
<https://mpi4py.scipy.org/>`_. Further, keep in mind that the ABCpy library has
to be properly installed on the cluster, such that it is available to the Python
interpreters on the master and the worker nodes.

Using the Spark Backend
~~~~~~~~~~~~~~~~~~~~~~~

To run ABCpy in parallel using Apache Spark, one only needs to use the provided
Spark backend. Considering the example from before, the statements for the
backend have to be changed to

.. literalinclude:: ../../examples/backends/apache_spark/pmcabc_gaussian.py
    :language: python
    :lines: 6-9
    :dedent: 4

In words, a Spark context has to be created and passed to the Spark
backend. Additionally, the level of parallelism can be provided, which defines in
a sense in how many blocks the work should be split up. It corresponds to the
parallelism of an RDD in Apache Spark terminology. A good value is usually a
small multiple of the total number of available cores.

The standard way to run the script on Spark is via the spark-submit command:

::

   PYSPARK_PYTHON=python3 spark-submit pmcabc_gaussian.py

Often Spark installations use Python 2 by default. To make Spark use the
required Python 3 interpreter, the `PYSPARK_PYTHON` environment variable can be
set.

The adapted python code can be found in
`examples/backend/apache_spark/pmcabc_gaussian.py`.

Note that in order to run jobs in parallel you need to have Apache Spark
installed on the system in question. The dependencies of the spark backend can be
install with `pip install -r requirements/backend-spark.txt`.

Details on the installation can be found on the official `homepage
<http://spark.apache.org>`_. Further, keep in mind that the ABCpy library has to
be properly installed on the cluster, such that it is available to the Python
interpreters on the master and the worker nodes.

Using Cluster Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When your model is computationally expensive and/or other factors require
compute infrastructure that goes beyond a single notebook or workstation you can
easily run ABCpy on infrastructure for cluster or high-performance computing.

Running on Amazon Web Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We show with high level steps how to get ABCpy running on Amazon Web Services
(AWS). Please note, that this is not a complete guide to AWS, so we would like
to refer you to the respective documentation. The first step would be to setup a
AWS Elastic Map Reduce (EMR) cluster which comes with the option of a
pre-configured Apache Spark. Then, we show how to run a simple inference code on
this cluster.

Setting up the EMR Cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~

When we setup an EMR cluster we want to install ABCpy on every node of the
cluster. Therefore, we provide a bootstrap script that does this job for us. On
your local machine create a file named `emr_bootstrap.sh` with the following
content:

::

   #!/bin/sh
   sudo yum -y install git
   sudo pip-3.4 install ipython findspark abcpy

In AWS go to Services, then S3 under the Storage Section. Create a new bucket
called `abcpy` and upload your bootstrap script `emr_bootstap.sh`.

To create a cluster, in AWS go to Services and then EMR under the Analytics
Section. Click 'Create Cluster', then choose 'Advanced Options'. In Step 1
choose the emr-5.7.0 image and make sure only Spark is selected for your cluster
(the other software packages are not required). In Step 2 choose for example one
master node and 4 core nodes (16 vCPUs if you have 4 vCPUs instances). In Step 3
under the boostrap action, choose custom, and select the script
`abcpy/emr_bootstrap.sh`. In the last step (Step 4), choose a key to access the
master node (we assume that you already setup keys). Start the cluster.


Running ABCpy on AWS
~~~~~~~~~~~~~~~~~~~~

Log in via SSH and run the following commands to get an example code from ABCpy
running with Python3 support:

::

   sudo bash -c 'echo export PYSPARK_PYTHON=python34 >> /etc/spark/conf/spark-env.sh'
   git clone https://github.com/eth-cscs/abcpy.git

Then, to submit a job to the Spark cluster we run the following commands:

::

   cd abcpy/examples/backends/
   spark-submit --num-executors 16 pmcabc_gaussian.py

Clearly the setup can be extended and optimized. For this and basic information
we refer you to the `AWS documentation on
EMR <http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-overview.html>`_.
