"""Module for the simulation kernel. Includes neutrino generation,
ray tracking (no raytracing yet), and hit generation."""

import logging
import numpy as np
from pyrex.internal_functions import normalize
from pyrex.signals import AskaryanSignal
from pyrex.ray_tracing import RayTracer
from pyrex.ice_model import IceModel

logger = logging.getLogger(__name__)


class EventKernel:
    """Kernel for generation of events with a given particle generator,
    list of antennas, and optionally a non-default ice_model."""
    def __init__(self, generator, antennas, ice_model=IceModel):
        self.gen = generator
        self.ice = ice_model
        self.antennas = antennas

    def event(self):
        """Generate particle, propagate signal through ice to antennas,
        process signal at antennas, and return the original particle."""
        p = self.gen.create_particle()
        logger.info("Processing event for %s", p)
        n = self.ice.index(p.vertex[2])
        for ant in self.antennas:
            rt = RayTracer(p.vertex, ant.position, ice_model=self.ice)

            # If no path(s) between the points, skip ahead
            if not rt.exists:
                logger.debug("Ray paths to %s do not exist", ant)
                continue

            for path in rt.solutions:
                # epol is (negative) vector rejection of
                # path.received_direction onto p.direction,
                # making epol orthogonal to path.recieved_direction in the same
                # plane as p.direction and path.received_direction
                epol = normalize(np.vdot(path.received_direction, p.direction)
                                 * path.received_direction - p.direction)
                # In case path.received_direction and p.direction are equal,
                # just let epol be all zeros

                psi = np.arccos(np.vdot(p.direction, path.emitted_direction))
                logger.debug("Angle to %s is %f degrees", ant, np.degrees(psi))
                # TODO: Support angles larger than pi/2
                # (low priority since these angles are far from cherenkov cone)
                if psi>np.pi/2:
                    continue

                # FIXME: Use shower energy for AskaryanSignal
                # Dependent on shower type / neutrino type

                times = np.linspace(-20e-9, 80e-9, 2000, endpoint=False)
                pulse = AskaryanSignal(times=times, energy=p.energy,
                                       theta=psi, n=n)

                path.propagate(pulse)
                # Dividing by path length scales Askaryan pulse properly
                pulse.values /= path.path_length

                ant.receive(pulse, direction=path.received_direction,
                            polarization=epol)

        return p
