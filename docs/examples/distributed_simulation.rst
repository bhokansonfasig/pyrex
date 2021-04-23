Distributed Effective Volume Simulation
=======================================

In this example we will illustrate how an effective volume simulation can be split across many jobs on a cluster that will each record their simulation in output files to be analyzed together after all the jobs are finished. This code can be found in the distributed_simulation.py script in the examples directory. A typical set of arguments for the script would look like the following:

.. code-block:: shell

    python distributed_simulation.py 1e9 2 --montecarlo -n 10000 -o my_output.h5

.. warning:: This script is intended to be run in many parallel instances across multiple jobs in a computing cluster in order to reach reasonable statistics at each energy. Once the jobs are run, more code would be necessary to combine and analyze the results. Such code is illustrated in the next example.

.. literalinclude:: code/distributed_simulation.py
    :language: python
