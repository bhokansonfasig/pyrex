
The IREX module contains classes for antennas and detectors which use waveform envelopes rather than raw waveforms. The detectors provided allow for testing of grid and station geometries.

.. currentmodule:: pryex.custom.irex

The :class:`EvelopeHpol` and :class:`EvelopeVpol` classes wrap models of ARA Hpol and Vpol antennas with an additional front-end which uses a diode-bridge circuit to create waveform envelopes. The trigger condition for these antennas is a simple threshold trigger on the envelopes.

The :class:`IREXString` class creates a string of :class:`EvelopeVpol` antennas at a given position. The :class:`RegularStation` class creates a station at a given position with 4 (or another given number) strings spaced evenly around the station center. The :class:`CoxeterStation` class creates a station at a given position similar to the :class:`RegularStation`, but with one string at the station center and the rest spaced evenly around the center. The :class:`StationGrid` class creates a rectangular grid of stations (or strings, as specified by the station type). The dimensions of the grid in stations is Nx by Ny where N is the total number of stations, Nx=floor(sqrt(N)), and Ny=floor(N/Nx).
