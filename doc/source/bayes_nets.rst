.. _bayes_nets:

Bayesian network - An introduction
======================================

Since ABCpy XX.XX, it is possible to perform inference on a Bayesian network. Here we consider an example of a Bayesian network from Wikipedia:

.. image:: bayes_net.png

The fact whether the grass is wet is our observed data. We can look outside and easily see whether this is true or not.
The observation of grass being wet can be due to two circumstances: the sprinkler is running or it is raining. This is indicated by the arrows going from sprinkler/rain to grass.
Finally, there is also an arrow going from rain to sprinkler. If there is rain, we assume our sprinkler is smart enough to not run.

In a Bayesian network, the random variables, corresponding to the unknown parameters (whether the sprinkler is running and whether it is raining) or the observed ones (whether the grass is wet), are called *nodes* and dependencies between them are represented as arrows (shown in the picture above).

As you can see, this structure can be seen as a graph. For a graph to be a Bayesian network, it needs to be a directed acyclic graph:

1) directed graph. This means that the arrows, the edges of the graph, have a direction associated with them. The direction tells us which random variable affects which other random variable. In our example, rain affects whether the sprinkler is running. Therefore, there is a directed edge from rain to sprinkler. However, the sprinkler does not affect whether it is raining, and, therefore, there is no directed edge from sprinkler to rain.

2) acyclic graph. This means that no random variable can indirectly or indirectly affect itself.

Now you can implement a similar dependency structure satisfying the two conditions above in ABCpy, and perform inference on it.
