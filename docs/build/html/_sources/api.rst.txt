.. _pyrex-api:

PyREx API
*********

The API documentation here is split into three sections.
First, the `api-1` section documents all classes and functions that are imported by PyREx under a ``from pyrex import *`` command.
Next, the `api-2` section is a full documentation of all the modules which make up the base PyREx package.
And finally, the `api-3` section documents the custom subpackages contained in PyREx by default.



.. _api-1:

Package contents
================

.. currentmodule:: pyrex

.. autoclass:: Signal
    :no-show-inheritance:

.. autoclass:: EmptySignal

.. autoclass:: FunctionSignal

.. autoclass:: AskaryanSignal

.. autoclass:: pyrex.signals.FastAskaryanSignal

.. autoclass:: ThermalNoise

.. autoclass:: Antenna
    :no-show-inheritance:

.. autoclass:: DipoleAntenna

.. autoclass:: AntennaSystem
    :no-show-inheritance:

.. autoclass:: Detector
    :no-show-inheritance:

.. autoclass:: IceModel
    :no-show-inheritance:

.. autoclass:: pyrex.ice_model.ArasimIce

.. autofunction:: prem_density

.. autofunction:: slant_depth

.. autoclass:: Particle
    :no-show-inheritance:

.. autoclass:: ShadowGenerator
    :no-show-inheritance:

.. autoclass:: RayTracer

.. autoclass:: pyrex.ray_tracing.SpecializedRayTracer

.. autoclass:: RayTracePath

.. autoclass:: pyrex.ray_tracing.SpecializedRayTracePath

.. autoclass:: EventKernel
    :no-show-inheritance:



.. _api-2:

Submodules
==========

pyrex\.signals module
---------------------

.. automodule:: pyrex.signals

pyrex\.antenna module
---------------------

.. automodule:: pyrex.antenna

pyrex\.detector module
----------------------

.. automodule:: pyrex.detector

pyrex\.ice\_model module
------------------------

.. automodule:: pyrex.ice_model

pyrex\.earth\_model module
--------------------------

.. automodule:: pyrex.earth_model

pyrex\.particle module
----------------------

.. automodule:: pyrex.particle

pyrex\.ray\_tracing module
--------------------------

.. automodule:: pyrex.ray_tracing

pyrex\.kernel module
--------------------

.. automodule:: pyrex.kernel

pyrex\.internal\_functions module
---------------------------------

.. automodule:: pyrex.internal_functions



.. _api-3:

PyREx Custom Subpackage
=======================

Note that more modules may be available as plug-ins, see `custom-package`.

pyrex\.custom\.pyspice module
-----------------------------

.. automodule:: pyrex.custom.pyspice

pyrex\.custom\.irex package
---------------------------

.. automodule:: pyrex.custom.irex

pyrex\.custom\.irex\.antenna module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyrex.custom.irex.antenna

pyrex\.custom\.irex\.detector module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyrex.custom.irex.detector

pyrex\.custom\.irex\.frontends module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyrex.custom.irex.frontends

pyrex\.custom\.irex\.reconstruction module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pyrex.custom.irex.reconstruction

