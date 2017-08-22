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
            pf = PathFinder(self.ice, p.vertex, ant.pos)
            if not(pf.exists):
                continue
            k = pf.emitted_ray
            epol = np.vdot(k, p.direction) * k - p.direction
            epol = epol / np.linalg.norm(epol)
            # p.direction and k should both be unit vectors
            psi = np.arccos(np.vdot(p.direction, k))

            times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)
            pulse = AskaryanSignal(times=times, energy=p.energy*1e-3,
                                   theta=psi, n=n)

            pf.propagate(pulse)

            ant.receive(pulse, epol)

        return p
