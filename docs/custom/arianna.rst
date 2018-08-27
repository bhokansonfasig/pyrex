
The ARIANNA module contains classes for antennas as found in the ARIANNA project.

.. currentmodule:: pyrex.custom.ARIANNA

The :class:`LPDA` class is the model of the ARIANNA LPDA antenna based on data from NuRadioReco. It uses directional/polarization gain from ``data/createLPDA_100MHz_InfFirn.ad1`` and ``createLPDA_100MHz_InfFirn.ra1``, and amplification gain from ``amp_300_gain.csv`` and ``amp_300_phase.csv``. The trigger condition of the antenna requires the signal to reach above and below some threshold values within a trigger window.
