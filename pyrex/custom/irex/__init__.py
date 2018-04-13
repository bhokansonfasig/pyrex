"""Customizations of pyrex package specific to IREX (IceCube Radio Extension)"""

from .antenna import IREXAntennaSystem
from .detector import (IREXGrid, IREXClusteredGrid, IREXCoxeterClusters,
                       IREXPairedGrid)
