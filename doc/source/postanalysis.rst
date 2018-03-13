.. _postanalysis:

5. Post Analysis
================

The output when sampling from an inference scheme is a Journal
(:py:class:`abcpy.output.Journal`) which holds all the necessary results and
convenient methods to do the post analysis.

For example, one can easily access the sampled parameters and corresponding
weights using:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 47-48
    :dedent: 4

Note the output format of `get_output_values()`: this will return a list. Depending on whether you specified that you would like full output during sampling or not, this list will have multiple entries or just one.

Each entry contains a Python dictionary. The keys for this dictionary are the names you specified for the parameters. The corresponding values are the inferred values for that parameter. Here is a short example of what you would specify, and what would be the output in the end:

.. code-block:: python

    a = Normal([[1],[0.1]], name='parameter_1')
    b = MultivariateNormal([[1,1],[[0.1,0],[0,0.1]]], name='parameter_2')

If one defined a model with these two parameters as inputs, a number of samples per parameters of 2, and let the algorithm run for two steps, the following would be the output of `journal.get_output_values()` if one specified full output to be 1:

.. code-block:: python

    [{'parameter_1' : [[0.95],[0.97]], 'parameter_2': [[0.98,1.03],[1.06,0.92]]},{'parameter_1': [[1.07],[0.98]], 'parameter_2': [[0.99, 1.02],[1.04,1.03]]}]

Since this is a dictionary, you can also access the values for each step as:

.. code-block:: python

    journal.get_output_values()[step_number]["name"]


For the post analysis basic functions are provided:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 51-53
    :dedent: 4

Also, to ensure reproducibility, every journal stores the parameters of the
algorithm that created it:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 56
    :dedent: 4

And certainly, a journal can easily be saved to and loaded from disk:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 59, 62
    :dedent: 4
