.. _pyrex-api:

PyREx API
*********

The API documentation here is split into three sections.
First, the :ref:`api-1` section documents all classes and functions that are imported by PyREx under a ``from pyrex import *`` command.
Next, the :ref:`api-2` section is a full documentation of all the modules which make up the base PyREx package.
And finally, the :ref:`api-3` section documents the custom subpackages included with PyREx by default.

.. currentmodule:: pyrex

.. _api-1:

PyREx Package Imports
=====================

.. autosummary::
    :toctree: docstrings
    :nosignatures:

    Signal
    EmptySignal
    FunctionSignal
    AskaryanSignal
    ThermalNoise
    Antenna
    DipoleAntenna
    AntennaSystem
    Detector
    ice
    earth
    NeutrinoInteraction
    Particle
    Event
    CylindricalGenerator
    RectangularGenerator
    ListGenerator
    FileGenerator
    RayTracer
    RayTracePath
    EventKernel
    File


.. _api-2:

Individual Module APIs
======================

.. toctree::
    :maxdepth: 1

    api/internal_functions
    api/signals
    api/askaryan
    api/antenna
    api/detector
    api/earth_model
    api/ice_model
    api/ray_tracing
    api/particle
    api/generation
    api/kernel
    api/io


.. _api-3:

Included Custom Sub-Packages
============================

.. toctree::
    :maxdepth: 1
    :glob:

    api/custom-*
