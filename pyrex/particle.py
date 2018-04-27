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
    in GeV. Note that the x and y ranges are (-dx/2, dx/2) and (-dy/2, dy/2)
    while the z range is (-dz, 0)."""
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


class ListGenerator:
    """Class to generate neutrinos by simply pulling them from a list of
    Particle objects. By default returns to the start of the list once the end
    is reached, but can optionally fail after reaching the list's end."""
    def __init__(self, particles, loop=True):
        if isinstance(particles, Particle):
            self.particles = [particles]
        else:
            self.particles = particles
        self.loop = loop
        self._index = -1

    def create_particle(self):
        """Pulls next particle from the list."""
        self._index += 1
        if not self.loop and self._index>=len(self.particles):
            raise StopIteration("No more particles to be generated")
        return self.particles[self._index%len(self.particles)]


class FileGenerator:
    """Class to generate neutrinos by pulling their vertex, direction, and
    energy from a (list of) .npz file(s). Each file must have three arrays,
    containing the vertices, directions, and energies respectively so the first
    particle will have properties given by the first elements of these three
    arrays. Tries to smartly figure out which array is which based on their
    names, but if the arrays are unnamed, assumes they are in the order used
    above."""
    def __init__(self, files):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self._file_index = -1
        self._next_file()

    def _next_file(self):
        """Pulls the next file into memory."""
        self._file_index += 1
        self._index = -1
        self.vertices = None
        self.directions = None
        self.energies = None
        if self._file_index>=len(self.files):
            raise StopIteration("No more particles to be generated")
        with np.load(self.files[self._file_index]) as data:
            if 'arr_0' in data:
                self.vertices = data['arr_0']
                self.directions = data['arr_1']
                self.energies = data['arr_2']
                return
            for key, val in data.items():
                key = key.lower()
                if 'vert' in key:
                    self.vertices = val
                elif key.startswith('v'):
                    self.vertices = val
                if 'dir' in key:
                    self.directions = val
                elif key.startswith('d'):
                    self.directions = val
                if 'en' in key:
                    self.energies = val
                elif key.startswith('e'):
                    self.energies = val
        if (self.vertices is None or self.directions is None
                or self.energies is None):
            raise KeyError("Could not interpret data keys of file "+
                           str(self.files[self._file_index]))
        if (len(self.vertices)!=len(self.directions) or
                len(self.vertices)!=len(self.energies)):
            raise ValueError("Vertex, direction, and energy lists must all be"+
                             " the same length")

    def create_particle(self):
        """Pulls the next particle from the file(s)."""
        self._index += 1
        if self.vertices is None or self._index>=len(self.vertices):
            self._next_file()
            return self.create_particle()
        v = self.vertices[self._index]
        d = self.directions[self._index]
        e = self.energies[self._index]
        return Particle(vertex=v, direction=d, energy=e)
