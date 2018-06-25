"""
Customizations of pyrex package specific to ARA (Askaryan Radio Array).

"""

from .antenna import HpolAntenna, VpolAntenna
from .detector import (ARAString, PhasedArrayString, RegularStation,
                       AlbrechtStation, HexagonalGrid)
