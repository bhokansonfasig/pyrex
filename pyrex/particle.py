"""
Module for particles (neutrinos) and neutrino interactions in the ice.

Included in the module are Particle and NeutrinoInteraction classes, as
well as different particle generators.

"""

import logging
import numpy as np
from pyrex.internal_functions import normalize
import pyrex.earth_model as earth_model

logger = logging.getLogger(__name__)

AVOGADRO_NUMBER = 6.02e23

class NeutrinoInteraction:
    """
    Class for describing neutrino interaction attributes.

    Stores parameters used to describe cross section and interaction length of
    a specific neutrino interaction.

    Parameters
    ----------
    c : float
        Cross section energy coefficient.
    p : float
        Cross section energy exponent.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type
        Type of signal, representing the units of the values.
    ValueTypes : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    Notes
    -----
    Neutrino intractions based on the GQRS Ultrahigh-Energy Neutrino
    Interactions Paper [1]_.

    References
    ----------
    .. [1] NEUTRINO INTERACTIONS REFERENCE

    """
    def __init__(self, c, p):
        self.c = c
        self.p = p

    def cross_section(self, E):
        """
        Calculate the neutrino cross section at a given energy.

        Parameters
        ----------
        E : float
            Energy (GeV) of the neutrino.

        Returns
        -------
        float
            Cross section (cm^2) of the neutrino at energy `E`.

        """
        return self.c * E**self.p

    def interaction_length(self, E):
        """
        Calculate the neutrino interaction length at a given energy.

        Parameters
        ----------
        E : float
            Energy (GeV) of the neutrino.

        Returns
        -------
        float
            Interaction length (cm) in water equivalent for the neutrino at
            energy `E`.

        """
        return 1 / (AVOGADRO_NUMBER * self.cross_section(E))

# Neutrino interactions from GQRS Ultrahigh-Energy Neutrino Interactions 1995
# https://arxiv.org/pdf/hep-ph/9512364.pdf
CC_NU = NeutrinoInteraction(2.69E-36, 0.402)
NC_NU = NeutrinoInteraction(1.06e-36, 0.408)
CC_NUBAR = NeutrinoInteraction(2.53e-36, 0.404)
NC_NUBAR = NeutrinoInteraction(0.98e-36, 0.410)

class Particle:
    """
    Class for storing particle attributes.

    Parameters
    ----------
    vertex : array_like
        Vector position (m) of the particle.
    direction : array_like
        Vector direction of the particle's velocity.
    energy : float
        Energy (GeV) of the particle.

    Attributes
    ----------
    vertex : array_like
        Vector position (m) of the particle.
    direction : array_like
        (Unit) vector direction of the particle's velocity.
    energy : float
        Energy (GeV) of the particle.

    """
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
    """
    Generate an arbitrary cartesian unit vector.

    Returns
    -------
    array_like
        (Unit) vector with a uniformly distributed random direction.

    Notes
    -----
    Generates random vector direction by pulling from uniform distributions for
    -1<cos(theta)<1 and 0<phi<2*pi.

    """
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
    """
    Class to generate neutrino vertices with Earth shadowing.

    Generates neutrinos in a box with given width, length, and height. Accounts
    for Earth shadowing by comparing the neutrino interaction length to the
    material thickness of the Earth along the neutrino path, and rejecting
    particles which would interact before reaching the vertex. Note the subtle
    difference in x and y ranges compared to the z range.

    Parameters
    ----------
    dx : float
        Width of the ice volume in the x-direction. Neutrinos generated within
        (-`dx` / 2, `dx` / 2).
    dy : float
        Length of the ice volume in the y-direction. Neutrinos generated within
        (-`dy` / 2, `dy` / 2).
    dz : float
        Height of the ice volume in the z-direction. Neutrinos generated within
        (-`dz`, 0).
    energy : float or function
        Energy (GeV) of the neutrinos. If ``float``, all neutrinos have the
        same constant energy. If ``function``, neutrinos are generated with the
        energy returned by successive function calls.

    Attributes
    ----------
    dx : float
        Width of the ice volume in the x-direction. Neutrinos generated within
        (-`dx` / 2, `dx` / 2).
    dy : float
        Length of the ice volume in the y-direction. Neutrinos generated within
        (-`dy` / 2, `dy` / 2).
    dz : float
        Height of the ice volume in the z-direction. Neutrinos generated within
        (-`dz`, 0).
    energy_generator : function
        Function returning energy (GeV) of the neutrinos by successive function
        calls.
    count : int
        Number of neutrinos produced by the generator.

    See Also
    --------
    pyrex.slant_depth : Calculates the material thickness of a chord cutting
                        through Earth.

    """
    def __init__(self, dx, dy, dz, energy):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        if not callable(energy):
            try:
                e = float(energy)
            except TypeError:
                raise ValueError("energy_generator must be a function "+
                                 "or a number")
            else:
                energy = lambda: e
        self.energy_generator = energy
        self.count = 0

    def create_particle(self):
        """
        Generate a neutrino.

        Creates a neutrino with a random vertex in the volume, a random
        direction, and an energy based on the ``energy_generator``. Accounts
        for Earth shadowing by discarding particles that wouldn't make it to
        the vertex based on the Earth's thickness along their path.

        Returns
        -------
        Particle
            Random neutrino object not shadowed by the Earth.

        """
        vtx = np.random.uniform(low=(-self.dx/2, -self.dy/2, -self.dz),
                                high=(self.dx/2, self.dy/2, 0))
        u = random_direction()
        nadir = np.arccos(u[2])
        depth = -vtx[2]
        t = earth_model.slant_depth(nadir, depth)
        E = self.energy_generator()
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
    """
    Class to generate neutrino vertices from a list.

    Generates neutrinos by simply pulling them from a list of `Particle`
    objects. By default returns to the start of the list once the end is
    reached, but can optionally fail after reaching the list's end.

    Parameters
    ----------
    particles : Particle or list of Particle
        List of `Particle` objects to draw from. If only a single `Particle`
        object is given, creates a list of that particle alone.
    loop : boolean, optional
        Whether or not to return to the start of the list after throwing the
        last `Particle`. If ``False``, raises an error if trying to throw
        after the last `Particle`.

    Attributes
    ----------
    particles : list of Particle
        List to draw `Particle` objects from, sequentially.
    loop : boolean
        Whether or not to loop through the list more than once.

    """
    def __init__(self, particles, loop=True):
        if isinstance(particles, Particle):
            self.particles = [particles]
        else:
            self.particles = particles
        self.loop = loop
        self._index = -1

    def create_particle(self):
        """
        Generate a neutrino.

        Pulls the next `Particle` object from the class's list of particles.

        Returns
        -------
        Particle
            Next neutrino object in the list.

        Raises
        ------
        StopIteration
            If ``loop`` is ``False`` and the end of the list has been exceeded.

        """
        self._index += 1
        if not self.loop and self._index>=len(self.particles):
            raise StopIteration("No more particles to be generated")
        return self.particles[self._index%len(self.particles)]


class FileGenerator:
    """
    Class to generate neutrino vertices from numpy file(s).

    Generates neutrinos by pulling their vertex, direction, and energy from a
    (list of) .npz file(s). Each file must have three arrays, containing the
    vertices, directions, and energies respectively so the first particle will
    have properties given by the first elements of these three arrays. Tries to
    smartly figure out which array is which based on their names, but if the
    arrays are unnamed, assumes they are in the order used above.

    Parameters
    ----------
    files : str or list of str
        List of file names containing neutrino information. If only a single
        file name is provided, creates a list with that file alone.

    Attributes
    ----------
    files : list of str
        List of file names containing neutrino information.
    vertices : ndarray
        Array of neutrino vertices from the current file.
    directions : ndarray
        Array of neutrino directions from the current file.
    energies : ndarray
        Array of neutrino energies from the current file.

    """
    def __init__(self, files):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self._file_index = -1
        self._next_file()

    def _next_file(self):
        """
        Pulls the next file into memory.

        Reads in the next file from the ``files`` list and stores its vertices,
        directions, and energies. Tries to smartly figure out which array is
        which based on their names, but if the arrays are unnamed, assumes they
        are in the order used above.

        Raises
        ------
        KeyError
            If the keys of the numpy file could not be interpreted.
        ValueError
            If the arrays of vertices, directions, and energies are not of the
            same length.

        """
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
            else:
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
        """
        Generate a neutrino.

        Pulls the next `Particle` object from the file(s).

        Returns
        -------
        Particle
            Next neutrino object from the file(s).

        Raises
        ------
        StopIteration
            If the end of the last file in the file list has been reached.

        """
        self._index += 1
        if self.vertices is None or self._index>=len(self.vertices):
            self._next_file()
            return self.create_particle()
        return Particle(vertex=self.vertices[self._index],
                        direction=self.directions[self._index],
                        energy=self.energies[self._index])
