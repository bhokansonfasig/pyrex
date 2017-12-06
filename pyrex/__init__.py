"""A Python package for simulation of Askaryan pulses and radio antennas.

PyREx (\ **Py**\ thon package for an IceCube **R**\ adio **Ex**\ tension) is,
as its name suggests, a Python package designed to simulate the measurement of
Askaryan pulses via a radio antenna array around the IceCube South Pole
Neutrino Observatory.
The code is designed to be modular so that it can also be applied to other
askaryan radio antennas (e.g. the ARA and ARIANA collaborations)."""

import os
import os.path

from .signals import Signal, EmptySignal, FunctionSignal, AskaryanSignal, ThermalNoise
from .antenna import Antenna, DipoleAntenna
from .ice_model import IceModel
from .earth_model import prem_density, slant_depth
from .particle import Particle, ShadowGenerator
from .ray_tracing import PathFinder
from .kernel import EventKernel

__version__ = "1.2.0"


# Allow users to create their own (or borrow from others) modules that add to
# the custom package. These "plug-in" modules should be kept in a
# .pyrex-custom directory inside the user's home directory.
# Note that plug-in modules should NOT have an __init__.py or else things break.
# Example directory structure below:
#   /some/path/pyrex/
#   |-- __init__.py
#   |-- ... (other pyrex modules)
#   |-- custom/
#   |   |-- irex.py
#   |   |-- ... (other built-in custom modules)
#
#   ~/.pyrex-custom/
#   |-- ara-customization/
#   |   |-- custom/
#   |   |   |-- ara.py
#   |-- other-collaboration-customization/
#   |   |-- custom/
#   |   |   |-- collaboration_name.py
#   |-- ... (other plug-in modules following same directory structure)
#
#   ./pyrex-custom/
#   |-- non-global-customization/
#   |   |-- custom/
#   |   |   |-- my_personal_customization.py
#   |-- ... (other plug-in modules following same directory structure)

# This plug-in format is made possible by PEP 420 (Implicit Namespace Packages)
# Inspired by David Beazley's 2015 PyCon talk on Modules and Packages
# https://youtu.be/0oTh1CXRaQ0?t=1h25m45s (example at 1:25:45-1:36:00)


user_plugin_locations = [os.path.expanduser("~/.pyrex-custom"),
                         "./pyrex-custom"]

for user_plugin_directory in user_plugin_locations:
    if os.path.isdir(user_plugin_directory):
        for directory in sorted(os.listdir(user_plugin_directory)):
            directory_path = os.path.join(user_plugin_directory, directory)
            if os.path.isdir(directory_path):
                __path__.append(directory_path)
