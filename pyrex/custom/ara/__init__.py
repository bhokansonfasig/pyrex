"""
Customizations of pyrex package specific to ARA (Askaryan Radio Array).

"""

from .antenna import HpolAntenna, VpolAntenna
from .detector import (ARAString, PhasedArrayString, RegularStation,
                       PhasedArrayStation, HexagonalGrid)
from .stations import ARA01, ARA02, ARA03, ARA04, ARA05
