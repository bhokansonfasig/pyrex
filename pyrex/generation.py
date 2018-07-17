"""
Module for particle (neutrino) generators.

Generators are responsible for the input of events into the simulation.

"""

import logging
import numpy as np
import pyrex.earth_model as earth_model
from pyrex.particle import Event, Particle, NeutrinoInteraction

logger = logging.getLogger(__name__)


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
    flavor_ratio : array_like, optional
        Flavor ratio of neutrinos to be generated. Of the form [electron, muon,
        tau] neutrino fractions.
    interaction_model : optional
        Class to use to describe interactions of the generated particles.
        Should inherit from (or behave like) the base ``Interaction`` class.

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
    get_energy : function
        Function returning energy (GeV) of the neutrinos by successive function
        calls.
    ratio : ndarary
        (Normalized) flavor ratio of neutrinos to be generated. Of the form
        [electron, muon, tau] neutrino fractions.
    interaction_model : Interaction
        Class to use to describe interactions of the generated particles.
    count : int
        Number of neutrinos produced by the generator.

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.
    pyrex.slant_depth : Calculates the material thickness of a chord cutting
                        through Earth.

    """
    def __init__(self, dx, dy, dz, energy, flavor_ratio=(1,1,1),
                 interaction_model=NeutrinoInteraction):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        if not callable(energy):
            try:
                e = float(energy)
            except TypeError:
                raise ValueError("energy argument must be a function "+
                                 "or a number")
            else:
                energy = lambda: e
        self.get_energy = energy
        self.ratio = np.array(flavor_ratio)/np.sum(flavor_ratio)
        self.interaction_model = interaction_model
        self.count = 0

    def get_vertex(self):
        """
        Get the vertex of the next particle to be generated.

        Randomly generates a vertex uniformly distributed within the specified
        ice volume.

        Returns
        -------
        ndarray
            Vector vertex in the ice volume.

        """
        return np.random.uniform(low=(-self.dx/2, -self.dy/2, -self.dz),
                                 high=(self.dx/2, self.dy/2, 0))

    def get_direction(self):
        """
        Get the direction of the next particle to be generated.

        Randomly generates a cartesian unit vector uniformly distributed over
        the unit sphere.

        Returns
        -------
        ndarray
            (Unit) vector direction.

        Notes
        -----
        Generates random vector direction by pulling from uniform distributions
        for -1<cos(theta)<1 and 0<phi<2*pi.

        """
        cos_theta = np.random.random_sample()*2-1
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.random_sample() * 2*np.pi
        return np.array([sin_theta * np.cos(phi),
                         sin_theta * np.sin(phi),
                         cos_theta])

    def get_particle_type(self):
        """
        Get the particle type of the next particle to be generated.

        Randomly generates a neutrino flavor according to the flavor ratio of
        the generator, and chooses neutrino or antineutrino based on an
        assumption that the neutrinos are generated from proton-gamma
        interactions.

        Returns
        -------
        Particle.Type
            Enum value for the type of the particle.

        See Also
        --------
        pyrex.Particle : Class for storing particle attributes.

        Notes
        -----
        The neutrino/antineutrino choice is based on Section 3 of [1]_.

        References
        ----------
        .. [1] A. Bhattacharya et al, "The Glashow resonance at IceCube."
            JCAP **1110**, 017 (2011).

        """
        rand_flavor = np.random.rand()
        # Electron neutrinos
        if rand_flavor<self.ratio[0]:
            if np.random.rand()<0.78:
                return Particle.Type.electron_neutrino
            else:
                return Particle.Type.electron_antineutrino
        # Muon neutrinos
        elif rand_flavor<self.ratio[0]+self.ratio[1]:
            if np.random.rand()<0.61:
                return Particle.Type.muon_neutrino
            else:
                return Particle.Type.muon_antineutrino
        # Tau neutrinos
        else:
            if np.random.rand()<0.61:
                return Particle.Type.tau_neutrino
            else:
                return Particle.Type.tau_antineutrino

    def get_exit_points(self, particle):
        """
        Get the intersections of the particle path with the ice volume edges.

        For the given `particle`, calculates where its travel path intersects
        with the edges of the ice volume.

        Parameters
        ----------
        particle : Particle
            Particle traveling through the ice.

        Returns
        -------
        enter_point, exit_point : ndarray
            Vector points where the particle's path intersects with the edges
            of the ice volume.

        See Also
        --------
        pyrex.Particle : Class for storing particle attributes.

        """
        enter_point = None
        exit_point = None
        sides = ((-self.dx/2, self.dx/2),
                 (-self.dy/2, self.dy/2),
                 (-self.dz, 0))
        for count in range(6):
            coord = int(count/2)
            min_max = count%2
            if particle.direction[coord]==0:
                continue
            scale = ((sides[coord][min_max] - particle.vertex[coord]) /
                     particle.direction[coord])
            intersection = particle.vertex + particle.direction * scale
            valid = True
            for i, pair in enumerate(sides):
                if i==coord:
                    continue
                if intersection[i]<pair[0] or intersection[i]>pair[1]:
                    valid = False
            if valid:
                sign = 1 if min_max==1 else -1
                if sign*particle.direction[coord]<0:
                    enter_point = intersection
                else:
                    exit_point = intersection
            if enter_point is not None and exit_point is not None:
                return enter_point, exit_point
        raise ValueError("Could not determine exit points")

    def get_weight(self, particle):
        """
        Get the weighting to be applied to the particle.

        Calculates the weight of `particle` based on the probability that it
        interacts at its given vertex in the ice volume.

        Parameters
        ----------
        particle : Particle
            Particle to be weighted.

        Returns
        -------
        float
            Weight of the given `particle`.

        See Also
        --------
        pyrex.Particle : Class for storing particle attributes.

        """
        entry_point, exit_point = self.get_exit_points(particle)
        in_ice_vector = np.array(exit_point) - np.array(entry_point)
        in_ice_length = np.sqrt(np.sum(in_ice_vector**2))
        vertex_vector = particle.vertex - np.array(entry_point)
        travel_length = np.sqrt(np.sum(vertex_vector**2))
        # Convert cm water equivalent interaction length to meters in ice
        interaction_length = (particle.interaction.total_interaction_length
                              / 0.92 / 100)
        return (in_ice_length/interaction_length *
                np.exp(-travel_length/interaction_length))


    def create_event(self):
        """
        Generate a neutrino event in the ice volume.

        Creates a neutrino with a random vertex in the volume, a random
        direction, and an energy based on ``get_energy``. Particle type is
        randomly chosen, and its interaction type is also randomly chosen based
        on the branching ratio. Accounts for Earth shadowing by discarding
        particles that wouldn't make it to their vertex based on the Earth's
        thickness along their path. Weights the particles according to their
        probability of interacting in the ice at their vertex. Currently each
        `Event` returned consists of only a single `Particle`.

        Returns
        -------
        Event
            Random neutrino event not shadowed by the Earth.

        See Also
        --------
        pyrex.Event : Class for storing a tree of `Particle` objects
                      representing an event.
        pyrex.Particle : Class for storing particle attributes.

        """
        vtx = self.get_vertex()
        u = self.get_direction()
        E = self.get_energy()
        particle_id = self.get_particle_type()
        particle = Particle(particle_id=particle_id, vertex=vtx, direction=u,
                            energy=E, interaction_model=self.interaction_model)

        # Check whether the particle would survive travel through the Earth
        nadir = np.arccos(u[2])
        depth = -vtx[2]
        t = earth_model.slant_depth(nadir, depth)
        x = t / particle.interaction.total_interaction_length
        self.count += 1
        rand_exponential = np.random.exponential()
        if rand_exponential > x:
            particle.weight = self.get_weight(particle)
            logger.debug("Successfully created %s with interaction weight %d",
                         particle, particle.weight)
            return Event(particle)
        else:
            # Particle was shadowed by the earth. Try again
            logger.debug("Particle creation shadowed by the Earth")
            return self.create_event()


class ListGenerator:
    """
    Class to generate neutrino events from a list.

    Generates events by simply pulling them from a list of `Event` objects. By
    default returns to the start of the list once the end is reached, but can
    optionally fail after reaching the list's end.

    Parameters
    ----------
    events : Event, or list of Event
        List of `Event` objects to draw from. If only a single `Event` object
        is given, creates a list of that event alone.
    loop : boolean, optional
        Whether or not to return to the start of the list after throwing the
        last `Event`. If ``False``, raises an error if trying to throw after
        the last `Event`.

    Attributes
    ----------
    events : list of Event
        List to draw `Event` objects from, sequentially.
    loop : boolean
        Whether or not to loop through the list more than once.

    See Also
    --------
    pyrex.Event : Class for storing a tree of `Particle` objects
                  representing an event.
    pyrex.Particle : Class for storing particle attributes.

    """
    def __init__(self, events, loop=True):
        if isinstance(events, Event):
            self.events = [events]
        else:
            self.events = events
        self.loop = loop
        self._index = -1

    def create_event(self):
        """
        Generate a neutrino event.

        Pulls the next `Event` object from the class's list of events.

        Returns
        -------
        Event
            Next `Event` object in the list of events.

        See Also
        --------
        pyrex.Event : Class for storing a tree of `Particle` objects
                      representing an event.
        pyrex.Particle : Class for storing particle attributes.

        Raises
        ------
        StopIteration
            If ``loop`` is ``False`` and the end of the list has been exceeded.

        """
        self._index += 1
        if not self.loop and self._index>=len(self.events):
            raise StopIteration("No more events to be generated")
        return self.events[self._index%len(self.events)]


class FileGenerator:
    """
    Class to generate neutrino events from numpy file(s).

    Generates neutrinos by pulling their attributes from a (list of) .npz
    file(s). Each file must have four to six arrays, containing the id values,
    vertices, directions, energies, and optional interaction types and weights
    respectively so the first particle will have properties given by the first
    elements of these arrays. Tries to smartly figure out which array is which
    based on their names, but if the arrays are unnamed, assumes they are in
    the order used above.

    Parameters
    ----------
    files : str or list of str
        List of file names containing neutrino information. If only a single
        file name is provided, creates a list with that file alone.
    interaction_model : optional
        Class used to describe the interactions of the stored particles.

    Attributes
    ----------
    files : list of str
        List of file names containing neutrino information.
    ids : ndarray
        Array of particle id values from the current file.
    vertices : ndarray
        Array of neutrino vertices from the current file.
    directions : ndarray
        Array of neutrino directions from the current file.
    energies : ndarray
        Array of neutrino energies from the current file.
    interactions : ndarray
        Array of interaction types from the current file.
    weights : ndarray
        Array of neutrino weights from the current file.

    Warnings
    --------
    This generator only supports `Event` objects containing a single `Particle`
    object. There is currently no way to read from files where an `Event`
    contains mutliple `Particle` objects with some dependencies.

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.
    pyrex.Event : Class for storing a tree of `Particle` objects
                  representing an event.
    pyrex.Particle : Class for storing particle attributes.

    """
    def __init__(self, files, interaction_model=NeutrinoInteraction):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.interaction_model = interaction_model
        self._file_index = -1
        self._next_file()

    def _next_file(self):
        """
        Pulls the next file into memory.

        Reads in the next file from the ``files`` list and stores its vertices,
        directions, and energies. Tries to smartly figure out which array is
        which based on their names, but if the arrays are unnamed, assumes they
        are in the following order: vertices, directions, energies, interaction
        types, weights.

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
        self.ids = None
        self.vertices = None
        self.directions = None
        self.energies = None
        self.interactions = None
        self.weights = None
        if self._file_index>=len(self.files):
            raise StopIteration("No more events to be generated")
        with np.load(self.files[self._file_index]) as data:
            if 'arr_0' in data:
                self.ids = data['arr_0']
                self.vertices = data['arr_1']
                self.directions = data['arr_2']
                self.energies = data['arr_3']
                if 'arr_4' in data:
                    self.interactions = data['arr_4']
                if 'arr_5' in data:
                    self.weights = data['arr_5']
            else:
                for key, val in data.items():
                    key = key.lower()
                    if 'id' in key:
                        self.ids = val
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
                    if 'int' in key:
                        self.interactions = val
                    elif 'type' in key:
                        self.interactions = val
                    elif 'curr' in key:
                        self.interactions = val
                    if 'weight' in key:
                        self.weights = val
                    elif key.startswith('w'):
                        self.weights = val
        if (self.vertices is None or self.directions is None
                or self.energies is None):
            raise KeyError("Could not interpret data keys of file "+
                           str(self.files[self._file_index]))
        if (len(self.ids)!=len(self.vertices) or
                len(self.ids)!=len(self.directions) or
                len(self.ids)!=len(self.energies) or
                (self.interactions is not None
                 and len(self.ids)!=len(self.interactions)) or
                (self.weights is not None
                 and len(self.ids)!=len(self.weights))):
            raise ValueError("All input lists must all be the same length")

    def create_event(self):
        """
        Generate a neutrino.

        Pulls the next `Particle` object from the file(s) and places it into
        an `Event` by itself.

        Returns
        -------
        Event
            Next neutrino `Event` object from the file(s).

        Raises
        ------
        StopIteration
            If the end of the last file in the file list has been reached.

        See Also
        --------
        pyrex.Event : Class for storing a tree of `Particle` objects
                      representing an event.
        pyrex.Particle : Class for storing particle attributes.

        """
        self._index += 1
        if self.vertices is None or self._index>=len(self.vertices):
            self._next_file()
            return self.create_event()
        if self.interactions is None:
            interaction = None
        else:
            interaction = self.interactions[self._index]
        if self.weights is None:
            weight = 1
        else:
            weight = self.weights[self._index]
        p = Particle(particle_id=self.ids[self._index],
                     vertex=self.vertices[self._index],
                     direction=self.directions[self._index],
                     energy=self.energies[self._index],
                     interaction_model=self.interaction_model,
                     interaction_type=interaction,
                     weight=weight)
        return Event(p)
