"""Module for particles (namely neutrinos) and neutrino interactions in the ice.
Interactions include Earth shadowing (absorption) effect."""

import numpy as np
from pyrex.internal_functions import normalize
import pyrex.earth_model as earth_model

AVOGADRO_NUMBER = 6.02e23

class NeutrinoInteraction:
    """Class for neutrino interaction attributes."""
    def __init__(self, c, p):
        self.c = c
        self.p = p

    def cross_section(self, E):
        """Return the cross section at a given energy E (GeV)."""
        return self.c * E**self.p

    def interaction_length(self, E):
        """Return the interaction length at a given energy E (GeV)."""
        return 1 / (AVOGADRO_NUMBER * self.cross_section(E))

CC_NU = NeutrinoInteraction(2.69E-36, 0.402)
# FIXME: add other interactions

class Particle:
    """Class for storing particle attributes. Consists of a 3-D vertex (m),
    3-D direction vector (automatically normalized), and an energy (GeV)."""
    def __init__(self, vertex, direction, energy):
        self.vertex = np.array(vertex)
        self.direction = normalize(direction)
        self.energy = energy

def random_direction():
    """Generate an arbitrary 3D unit vector."""
    cos_theta = np.random.random_sample()*2-1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = np.random.random_sample() * 2*np.pi
    return [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]

    # Old method:
    # while 1:
    #     u = np.random.uniform(low=(-1,-1,-1), high=(1,1,1))
    #     mag = np.linalg.norm(u)
    #     if mag<1.0:
    #         s = 1.0/mag
    #         return u * s

class ShadowGenerator:
    """Class to generate UHE neutrino vertices in (relatively) shallow
    detectors. Takes into accout Earth shadowing (sort of).
    energy_generator should be a function that returns a particle energy
    in GeV."""
    # TODO: Properly account for NC and anti-neutrino interactions
    # Currently the cross section is just the CC cross section
    def __init__(self, dx, dy, dz, energy_generator):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        if not callable(energy_generator):
            raise ValueError("energy_generator must be a function")
        self.egen = energy_generator
        self.count = 0

    def create_particle(self):
        """Creates a particle with random vertex in cube with a random
        direction."""
        vtx = np.random.uniform(low=(-self.dx/2, -self.dy/2, -self.dz),
                                high=(self.dx/2, self.dy/2, 0))
        u = random_direction()
        nadir = np.arccos(u[2])
        depth = -vtx[2]
        t = earth_model.slant_depth(nadir, depth)
        E = self.egen()
        # FIXME: Add other interactions
        inter_length = CC_NU.interaction_length(E)
        x = t / inter_length
        self.count += 1
        rand_exponential = np.random.exponential()
        if rand_exponential > x:
            return Particle(vtx, u, E)
        else:
            return self.create_particle()
