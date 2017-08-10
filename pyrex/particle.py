"""Module for particles (namely neutrinos) and neutrino interactions in the ice.
Interactions include Earth shadowing (absorption) effect."""

from collections import namedtuple
import numpy as np
import pyrex.earth_model as earth_model

AVOGADRO_NUMBER = 6.02e23

class NeutrinoInteraction:
    """Class for neutrino interaction attributes."""
    def __init__(self, c, p):
        self.c = c
        self.p = p

    def cross_section(self, E):
        """Return the cross section at a given energy E."""
        return (self.c * E)**self.p

    def interaction_length(self, E):
        """Return the interaction length at a given energy E."""
        return 1 / (AVOGADRO_NUMBER * self.cross_section(E))

CC_NU = NeutrinoInteraction(2.69E-36, 0.402)
# FIXME: add other interactions

Particle = namedtuple('Particle', ['vertex','direction','energy'])

def next_direction():
    """Generate an arbitrary 3D unit vector"""
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
    detectors. Takes into accout Earth shadowing (sort of)."""
    # TODO: Properly account for NC and anti-neutrino interactions
    # Currently the cross section is just the CC cross section
    def __init__(self, dx, dy, dz, egen):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.egen = egen
        self.n = 0

    def create_particle(self):
        vtx = np.random.uniform(low=(-self.dx/2, -self.dy/2, -self.dz),
                                high=(self.dx/2, self.dy/2, 0))
        u = next_direction()
        nadir = np.arccos(u[2])
        depth = -vtx[2]
        E = self.egen()
        t = earth_model.slant_depth(nadir, depth)
        # FIXME: Add other interactions
        ilen = CC_NU.interaction_length(E)
        x = t / ilen
        self.n += 1
        expRV = np.random.exponential()
        if expRV > x:
            return Particle(vtx, u, E)
        else:
            return self.create_particle()
