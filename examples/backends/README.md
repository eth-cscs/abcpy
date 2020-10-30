# Parallelization Backends  
We showcase here how to use the different parallelization backends with the same inference problem. See [here](https://abcpy.readthedocs.io/en/latest/parallelization.html#) for more information.

## Apache Spark

This uses the Apache Spark backend for parallelization. It relies on the `pyspark` and `findspark` library. 

In this setup, the number of parallel processes is defined inside the Python code, with the following lines:

    import pyspark
    sc = pyspark.SparkContext()
    from abcpy.backends import BackendSpark as Backend
    backend = Backend(sc, parallelism=4)

Then, the parallel script can be run with: 

    PYSPARK_PYTHON=python3 spark-submit apache_spark/pmcabc_gaussian.py 

where the environment variable `PYSPARK_PYTHON` is set as often Spark installations use Python2 by default.    


## Dummy

This is a dummy backend which does not parallelize; it is useful for debug and testing purposes. Simply run the Python file as normal.

## MPI

This used MPI to distribute the inference task; we exploit the `mpi4py` Python library for using MPI from Python.

Mainly, we distribute data generation from the model, which is usually the most expensive part in ABC inference. 

We have two files in `mpi` folder:
1. `pmcabc_gaussian.py` performs a simple inference experiment on a gaussian model with PMCABC; this is the same as in the other two backends
2. `mpi_model_inferences.py` showcases how to use nested MPI parallelization with a model which already has some level of parallelization with MPI. That is done with several ABC algorithms. See below to understand how to run this file correctly.

To run the files with MPI, the following command is required:

    mpirun -n <n_tasks> python3 <filename.py>

For instance, to run `pmcabc_gaussian.py` with 4 tasks, we can run: 

    mpirun -n 4 python3 mpi/pmcabc_gaussian.py

### Nested parallelization with MPI

In `mpi_model_inferences.py`, the model itself is parallelized with MPI. We can run nested parallelized inference by considering _n_ independent model instances (ie we simulate _n_ independent copies of the model at once) each of which is assigned _m_ MPI tasks. Moreover, we also require one additional MPI task to work as a master in this setup. Therefore, in total we need _(n * m) + 1_ MPI tasks. In this case, we have set _m=2_ in the Python code via the lines: 

```
from abcpy.backends import BackendMPI as Backend
backend = Backend(process_per_model=2)
```

Let's say we want to parallelize the model _n=3_ times. Therefore, we use the following command:

    mpirun -n 7 python3 mpi/mpi_model_inferences.py

as _(3*2) + 1 = 7_. Note that, in this scenario, using only 6 tasks overall leads to failure of the script due to how the tasks are assigned to the model instances.  
