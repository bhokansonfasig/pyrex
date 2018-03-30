"""Module for particles (namely neutrinos) and neutrino interactions in the ice.
Interactions include Earth shadowing (absorption) effect."""

import logging
import numpy as np
from pyrex.internal_functions import normalize
import pyrex.earth_model as earth_model

logger = logging.getLogger(__name__)

AVOGADRO_NUMBER = 6.02e23

class NeutrinoInteraction:
    """Class for neutrino interaction attributes."""
    def __init__(self, c, p):
        self.c = c
        self.p = p

    def cross_section(self, E):
        """Return the cross section (cm^2) at a given energy E (GeV)."""
        return self.c * E**self.p

    def interaction_length(self, E):
        """Return the interaction length (cm) in water equivalent at a
        given energy E (GeV)."""
        return 1 / (AVOGADRO_NUMBER * self.cross_section(E))

# Neutrino interactions from GQRS Ultrahigh-Energy Neutrino Interactions 1995
# https://arxiv.org/pdf/hep-ph/9512364.pdf
CC_NU = NeutrinoInteraction(2.69E-36, 0.402)
NC_NU = NeutrinoInteraction(1.06e-36, 0.408)
CC_NUBAR = NeutrinoInteraction(2.53e-36, 0.404)
NC_NUBAR = NeutrinoInteraction(0.98e-36, 0.410)

class Particle:
    """Class for storing particle attributes. Consists of a 3-D vertex (m),
    3-D direction vector (automatically normalized), and an energy (GeV)."""
    def __init__(self, vertex, direction, energy):
        self.vertex = np.array(vertex)
        self.direction = normalize(direction)
        self.energy = energy

    def __str__(self):
        string = self.__class__.__name__+"("
        for key, val in self.__dict__.items():
            string += key+"="+repr(val)+", "
        return string[:-2]+")"

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
        # Interaction length is average of neutrino and antineutrino
        # interaction lengths. Each of those is the inverted-sum of the
        # CC and NC interaction lengths.
        inter_length = 2/(1/CC_NU.interaction_length(E) +
                          1/NC_NU.interaction_length(E) +
                          1/CC_NUBAR.interaction_length(E) +
                          1/NC_NUBAR.interaction_length(E))
        x = t / inter_length
        self.count += 1
        rand_exponential = np.random.exponential()
        if rand_exponential > x:
            p = Particle(vtx, u, E)
            logger.debug("Successfully created %s", p)
            return p
        else:
            # Particle was shadowed by the earth. Try again
            logger.debug("Particle creation shadowed by the Earth")
            return self.create_particle()
