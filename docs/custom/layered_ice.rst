
The layered ice module contains custom ice models and ray tracers needed for handling stratified ice layers.

.. currentmodule:: pyrex.custom.layered_ice

The :class:`LayeredIce` class is used to define ice models for individual layers of ice, as well as the depth ranges over which these ice models are to be applied. The :class:`LayeredRayTracer` class then takes two endpoints and a :class:`LayeredIce` instance and returns all valid :class:`LayeredRayTracePath` objects between those two endpoints. A maximum number of reflections is allowed between the layer boundaries, as specified by :attr:`LayeredRayTracer.max_reflections`.
