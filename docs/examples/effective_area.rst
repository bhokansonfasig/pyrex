Calculate Effective Area
========================

In this example we will calculate the effective area of a detector over a range of energies. This code can be run from the `effective_area.py <https://github.com/bhokansonfasig/pyrex/blob/master/examples/effective_area.py>`_ script in the examples directory.

.. warning:: In order to finish reasonably quickly, the number of events thrown in this example is low. This means that there are likely not enough events to accurately represent the effective area of the detector. For an accurate measurement, the number of events must be increased, but this will need much more time to run in that case. For the plots below, a higher number of events was thrown.

.. literalinclude:: code/effective_area.py
    :language: python

.. image:: ../_static/example_outputs/effective_area_1.png
.. image:: ../_static/example_outputs/effective_area_2.png
