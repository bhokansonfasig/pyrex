"""Module containing setup and wrappers for PySpice module into PyREx"""

import importlib.util
import numpy as np

# Check if PySpice can be imported on the current system
# This variable can be checked before using PySpice in other modules
__available__ = importlib.util.find_spec('PySpice') is not None

__modulenotfound__ = ("PySpice could not be imported. "+
                      "For details on installing PySpice, see "+
                      "https://pyspice.fabrice-salvaire.fr/installation.html")

if __available__:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Spice.Library import SpiceLibrary
    from PySpice.Unit import *

    class NgSpiceSharedSignal(NgSpiceShared):
        """Helper class for bridging gap between PyREx Signal and
        PySpice NgSpiceShared classes"""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._signal = None

        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
            voltage[0] = np.interp(time, self._signal.times, self._signal.values)
            return 0

    NGSPICE_SHARED_MASTER = NgSpiceSharedSignal()

    class SpiceSignal:
        """Class for passing PyREx Signal object into PySpice"""
        def __init__(self, signal, shared=NGSPICE_SHARED_MASTER):
            self.shared = shared
            self.shared._signal = signal
