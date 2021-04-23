
The ARA module contains classes for antennas and detectors as found or proposed for the ARA project.

.. currentmodule:: pyrex.custom.ara

The :class:`HpolAntenna` and :class:`VpolAntenna` classes are models of ARA Hpol and Vpol antennas using data lifted from AraSim. They use the antenna directional gains in ``data/Hpol_original_150mmHole_Ice_ARASim.txt`` and ``data/Vpol_original_CrossFeed_150mmHole_Ice_ARASim.txt`` respectively, and the electronics gains in ``data/ARA_Electronics_TotalGain_TwoFilters.txt``. The trigger condition of these antennas is based on a comparison of the maximum value of the tunnel-diode-convolved waveforms with the rms value of a tunnel-diode-convolved noise waveform.

The :class:`ARAString` class creates a string of alternating :class:`HpolAntenna` and :class:`VpolAntenna` objects, as in a typical ARA station. The :class:`PhasedArrayString` class implements a more densely-packed string of antennas which trigger based on a threshold trigger on the best beam-formed combination of the antenna waveforms. The :class:`RegularStation` class creates a station at the given position with 4 (or another given number) strings spaced evenly around the station center. The :class:`PhasedArrayStation` class creates station with a phased array string at the station center, and 4 (or another given number) outrigger strings spaced evenly around the station center. The :class:`HexagonalGrid` class creates a hexagonal grid of stations, spiralling outward from the center.
