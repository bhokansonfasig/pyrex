"""
PyREx radio neutrino simulation package.

"""

import os
import os.path

from .__about__ import __version__, __long_description__
__doc__ = __long_description__

from .signals import Signal, EmptySignal, FunctionSignal, ThermalNoise
from .askaryan import AskaryanSignal
from .antenna import Antenna, DipoleAntenna
from .detector import AntennaSystem, Detector
from .ice_model import ice
from .earth_model import earth
from .particle import Event, Particle, NeutrinoInteraction
from .generation import (CylindricalGenerator, RectangularGenerator,
                         ListGenerator, FileGenerator)
from .ray_tracing import RayTracer, RayTracePath
from .kernel import EventKernel
from .io import File


# Allow users to create their own (or borrow from others) modules that add to
# the custom package. These "plug-in" modules should be kept in a
# .pyrex-custom directory inside the user's home directory.
# Note that plug-in "custom" directories should NOT have an __init__.py
# or else things break.
# Example directory structure below:
#   /some/path/pyrex/
#   |-- __init__.py
#   |-- ... (other pyrex modules)
#   |-- custom/
#   |   |-- pyspice.py
#   |   |-- irex/
#   |   |   |-- __init__.py
#   |   |   |-- antenna.py
#   |   |   |-- ... (other irex-related modules)
#   |   |-- ... (other built-in custom modules/packages)
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
