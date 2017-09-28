"""Module for the simulation kernel. Includes neutrino generation,
ray tracking (no raytracing yet), and hit generation."""

import numpy as np
import scipy.fftpack
from pyrex.signals import Signal, AskaryanSignal
from pyrex.ray_tracing import PathFinder


class EventKernel:
    """Kernel for generation of events with a given particle generator,
    ice model, and list of antennas."""
    def __init__(self, generator, ice_model, antennas):
        self.gen = generator
        self.ice = ice_model
        self.ant_array = antennas

    def event(self):
        """Generate particle, propagate signal through ice to antennas,
        process signal at antennas, and return the original particle."""
        p = self.gen.create_particle()
        n = self.ice.index(p.vertex[2])
        for ant in self.ant_array:
            pf = PathFinder(self.ice, p.vertex, ant.position)
            if not(pf.exists):
                continue

            # p.direction and k should both be unit vectors
            # epol is (negative) vector rejection of k onto p.direction
            k = pf.emitted_ray
            epol = np.vdot(k, p.direction) * k - p.direction
            # In case k and p.direction are equal
            # (antenna directly on shower axis), just let epol be all zeros
            # Don't divide so no divide-by-zero warning is thrown
            if not(np.all(epol==0)):
                epol = epol / np.linalg.norm(epol)

            psi = np.arccos(np.vdot(p.direction, k))

            # TODO: Support angles larger than pi/2
            if psi>np.pi/2:
                continue

            times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)
            pulse = AskaryanSignal(times=times, energy=p.energy*1e-3,
                                   theta=psi, n=n)

            pf.propagate(pulse)
            # Dividing by path length scales Askaryan pulse properly
            pulse.values /= pf.path_length

            ant.receive(pulse, epol)

        return p
