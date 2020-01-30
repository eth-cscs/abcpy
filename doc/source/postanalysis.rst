.. _postanalysis:

5. Post Analysis
================

The output of an inference scheme is a Journal
(:py:class:`abcpy.output.Journal`) which holds all the necessary results and
convenient methods to do the post analysis.

For example, one can easily access the sampled parameters and corresponding
weights using:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 48-49
    :dedent: 4

The output of `get_parameters()` is a Python dictionary. The keys for this dictionary are the names you specified for the parameters. The corresponding values are the marginal posterior samples of that parameter. Here is a short example of what you would specify, and what would be the output in the end:

.. code-block:: python

    a = Normal([[1],[0.1]], name='parameter_1')
    b = MultivariateNormal([[1,1],[[0.1,0],[0,0.1]]], name='parameter_2')

If one defined a model with these two parameters as inputs and `n_sample=2`, the following would be the output of `journal.get_parameters()`:

.. code-block:: python

    {'parameter_1' : [[0.95],[0.97]], 'parameter_2': [[0.98,1.03],[1.06,0.92]]}

These are samples at the final step of ABC algorithm. If you want samples from the earlier steps you can get a Python dictionary for that step by using:

.. code-block:: python

    journal.get_parameters(step_number)

Since this is a dictionary, you can also access the values for each step as:

.. code-block:: python

    journal.get_parameters(step_number)["name"]


For the post analysis basic functions are provided:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 51-54
    :dedent: 4

Also, to ensure reproducibility, every journal stores the parameters of the
algorithm that created it:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 57
    :dedent: 4

Finally, you can plot the inferred posterior mean of the parameters in the following way:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 65
    :dedent: 4

And certainly, a journal can easily be saved to and loaded from disk:

.. literalinclude:: ../../examples/backends/dummy/pmcabc_gaussian.py
    :language: python
    :lines: 60, 63
    :dedent: 4
