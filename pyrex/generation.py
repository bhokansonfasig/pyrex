"""
Module for particle (neutrino) generators.

Generators are responsible for the input of events into the simulation.

"""

from collections.abc import Iterable
from enum import Enum
import logging
import numpy as np
from pyrex.internal_functions import get_from_enum
from pyrex.earth_model import earth
from pyrex.particle import Event, Particle, NeutrinoInteraction
from pyrex.io import File

logger = logging.getLogger(__name__)


class Generator:
    """
    Base class for neutrino generators.

    Provides methods for generating neutrino attributes except for neutrino
    vertex, which should be provided by child classes to generate neutrinos
    in specific volumes.

    Parameters
    ----------
    energy : float or function
        Energy (GeV) of the neutrinos. If ``float``, all neutrinos have the
        same constant energy. If ``function``, neutrinos are generated with the
        energy returned by successive function calls.
    shadow : bool, optional
        Whether Earth shadowing effects should be used to reject events. If
        ``True`` then neutrinos which don't survive transit through the Earth
        will be skipped when creating events. If ``False`` then all events are
        allowed and assigned a weight to scale their probability of occurrence.
    flavor_ratio : array_like, optional
        Flavor ratio of neutrinos to be generated. Of the form [electron, muon,
        tau] neutrino fractions.
    source : optional
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : optional
        Class to use to describe interactions of the generated particles.
        Should inherit from (or behave like) the base ``Interaction`` class.

    Attributes
    ----------
    count : int
        Number of neutrinos produced by the generator, including those not
        returned due to Earth shadowing or other effects.
    get_energy : function
        Function returning energy (GeV) of the neutrinos by successive function
        calls.
    shadow : bool
        Whether Earth shadowing effects will be used to reject events.
    ratio : ndarray
        (Normalized) flavor ratio of neutrinos to be generated. Of the form
        [electron, muon, tau] neutrino fractions.
    source : Generator.SourceType
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : Interaction
        Class to use to describe interactions of the generated particles.
    volume
    solid_angle

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.

    """
    class SourceType(Enum):
        """
        Enum containing possible sources for neutrinos.

        Attributes
        ----------
        pgamma, cosmogenic
        pp, astrophysical
        unknown, undefined

        """
        undefined = 0
        unknown = 0
        cosmogenic = 1
        pgamma = 1
        astrophysical = 2
        pp = 2

    def __init__(self, energy, shadow=False, flavor_ratio=(1,1,1),
                 source="cosmogenic", interaction_model=NeutrinoInteraction,
                 earth_model=earth):
        if not callable(energy):
            try:
                e = float(energy)
            except TypeError:
                raise ValueError("energy argument must be a function "+
                                 "or a number")
            else:
                energy = lambda: e
        self.get_energy = energy
        self.shadow = shadow
        self.ratio = np.array(flavor_ratio)/np.sum(flavor_ratio)
        self.source = source
        self.interaction_model = interaction_model
        self.earth_model = earth_model
        self.count = 0

    @property
    def source(self):
        """
        Value of the source type.

        Should always be a value from the ``Interaction.Type`` enum. Setting
        with integer or string values may work if carefully chosen.

        """
        return self._source

    @source.setter
    def source(self, src_type):
        if src_type is None:
            self._source = self.SourceType.undefined
        else:
            self._source = get_from_enum(src_type, self.SourceType)

    @property
    def volume(self):
        """
        Generation volume (m^3) in which event vertices are produced.

        """
        raise NotImplementedError("volume property must be implemented by "+
                                  "inheriting class")

    @property
    def solid_angle(self):
        """
        Generation solid angle (sr) in which event directions are produced.

        """
        logger.debug("Using default solid_angle from "+
                     "pyrex.generation.Generator")
        return 4 * np.pi

    def get_vertex(self):
        """
        Get the vertex of the next particle to be generated.

        For the `Generator` class, this method is not implemented.
        Subclasses should override this method with their own procedure for
        generating neutrino vertices in some volume.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides the function.

        """
        logger.debug("Using default get_vertex from "+
                     "pyrex.generation.Generator")
        raise NotImplementedError("get_vertex method must be implemented by "
                                  +"inheriting class")

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
        the generator, and chooses neutrino or antineutrino based on ratios
        derived from the source type.

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
            JCAP **1110**, 017 (2011). :arxiv:`1108.3163`
            :doi:`10.1088/1475-7516/2011/10/017`

        """
        rand_flavor = np.random.rand()
        rand_nunubar = np.random.rand()
        if self.source==self.SourceType.cosmogenic:
            nunubar_ratios = [0.78, 0.61, 0.61]
        elif self.source==self.SourceType.astrophysical:
            nunubar_ratios = [0.5, 0.5, 0.5]
        else:
            raise ValueError("Source type not supported")

        # Electron neutrinos
        if rand_flavor<self.ratio[0]:
            if rand_nunubar<nunubar_ratios[0]:
                return Particle.Type.electron_neutrino
            else:
                return Particle.Type.electron_antineutrino
        # Muon neutrinos
        elif rand_flavor<self.ratio[0]+self.ratio[1]:
            if rand_nunubar<nunubar_ratios[1]:
                return Particle.Type.muon_neutrino
            else:
                return Particle.Type.muon_antineutrino
        # Tau neutrinos
        else:
            if rand_nunubar<nunubar_ratios[2]:
                return Particle.Type.tau_neutrino
            else:
                return Particle.Type.tau_antineutrino

    def get_exit_points(self, particle):
        """
        Get the intersections of the particle path with the ice volume edges.

        For the `Generator` class, this method is not implemented.
        Subclasses should override this method with their own procedure for
        calculating exit points given the generation volume.

        Parameters
        ----------
        particle : Particle
            Particle traveling through the ice.

        Raises
        ------
        NotImplementedError
            Always, unless a subclass overrides the function.

        See Also
        --------
        pyrex.Particle : Class for storing particle attributes.

        """
        logger.debug("Using default get_exit_points from "+
                     "pyrex.generation.Generator")
        raise NotImplementedError("get_exit_points method must be implemented "
                                  +"by inheriting class")

    def get_weights(self, particle):
        """
        Get the weighting factors to be applied to the particle.

        Calculates both the survival and interaction weights of `particle`.
        The survival weight is based on the probability of interaction along
        the path through the Earth. The interaction weight of `particle` based
        on the probability of interaction at its given vertex in the ice
        volume.

        Parameters
        ----------
        particle : Particle
            Particle to be weighted.

        Returns
        -------
        survival_weight : float
            Survival weight of the given `particle`.
        interaction_weight : float
            Interaction weight of the given `particle`.

        See Also
        --------
        pyrex.Particle : Class for storing particle attributes.

        """
        t = self.earth_model.slant_depth(particle.vertex, -particle.direction)
        x = t / particle.interaction.total_interaction_length
        survival_weight = np.exp(-x)

        entry_point, exit_point = self.get_exit_points(particle)
        in_ice_vector = np.array(exit_point) - np.array(entry_point)
        in_ice_length = np.sqrt(np.sum(in_ice_vector**2))
        vertex_vector = particle.vertex - np.array(entry_point)
        travel_length = np.sqrt(np.sum(vertex_vector**2))
        # Convert cm water equivalent interaction length to meters in ice
        interaction_length = (particle.interaction.total_interaction_length
                              / 0.92 / 100)
        interaction_weight = (in_ice_length/interaction_length *
                              np.exp(-travel_length/interaction_length))

        return survival_weight, interaction_weight


    def create_event(self):
        """
        Generate a neutrino event in the ice volume.

        Creates a neutrino with a random vertex in the volume, a random
        direction, and an energy based on ``get_energy``. Particle type is
        randomly chosen, and its interaction type is also randomly chosen based
        on the branching ratio. Weights the particles according to their
        survival probability through the Earth and their probability of
        interacting in the ice at their vertex. If Earth shadowing has been
        turned on then particles which don't survive transit through the Earth
        are skipped, and surviving particles are given a survival weight of 1.
        Currently each `Event` returned consists of only a single `Particle`.

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
        self.count += 1
        vtx = self.get_vertex()
        u = self.get_direction()
        E = self.get_energy()
        particle_id = self.get_particle_type()
        particle = Particle(particle_id=particle_id, vertex=vtx, direction=u,
                            energy=E, interaction_model=self.interaction_model)

        weights = self.get_weights(particle)
        if not self.shadow:
            particle.survival_weight = weights[0]
            particle.interaction_weight = weights[1]
            logger.debug("Successfully created %s with survival weight %d and "
                         +"interaction weight %d", particle, weights[0],
                         weights[1])
            return Event(particle)
        elif np.random.rand() < weights[0]:
            particle.survival_weight = 1
            particle.interaction_weight = weights[1]
            logger.debug("Successfully created %s with survival weight %d and "
                         +"interaction weight %d", particle, weights[0],
                         weights[1])
            return Event(particle)
        else:
            # Particle was shadowed by the earth. Try again
            logger.debug("Particle creation shadowed by the Earth")
            return self.create_event()


class CylindricalGenerator(Generator):
    """
    Class to generate neutrino vertices in a cylindrical ice volume.

    Generates neutrinos in a cylinder with given radius and height.

    Parameters
    ----------
    dr : float
        Radius of the ice volume. Neutrinos generated within (0, `dr`).
    dz : float
        Height of the ice volume in the z-direction. Neutrinos generated within
        (-`dz`, 0).
    energy : float or function
        Energy (GeV) of the neutrinos. If ``float``, all neutrinos have the
        same constant energy. If ``function``, neutrinos are generated with the
        energy returned by successive function calls.
    shadow : bool, optional
        Whether Earth shadowing effects should be used to reject events. If
        ``True`` then neutrinos which don't survive transit through the Earth
        will be skipped when creating events. If ``False`` then all events are
        allowed and assigned a weight to scale their probability of occurrence.
    flavor_ratio : array_like, optional
        Flavor ratio of neutrinos to be generated. Of the form [electron, muon,
        tau] neutrino fractions.
    source : optional
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : optional
        Class to use to describe interactions of the generated particles.
        Should inherit from (or behave like) the base ``Interaction`` class.

    Attributes
    ----------
    count : int
        Number of neutrinos produced by the generator, including those not
        returned due to Earth shadowing or other effects.
    dr : float
        Radius of the ice volume. Neutrinos generated within (0, `dr`).
    dz : float
        Height of the ice volume in the z-direction. Neutrinos generated within
        (-`dz`, 0).
    get_energy : function
        Function returning energy (GeV) of the neutrinos by successive function
        calls.
    shadow : bool
        Whether Earth shadowing effects will be used to reject events.
    ratio : ndarray
        (Normalized) flavor ratio of neutrinos to be generated. Of the form
        [electron, muon, tau] neutrino fractions.
    source : Generator.SourceType
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : Interaction
        Class to use to describe interactions of the generated particles.
    volume
    solid_angle

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.

    """
    def __init__(self, dr, dz, energy, shadow=False, flavor_ratio=(1,1,1),
                 source="cosmogenic", interaction_model=NeutrinoInteraction,
                 earth_model=earth):
        self.dr = dr
        self.dz = dz
        super().__init__(energy=energy, shadow=shadow,
                         flavor_ratio=flavor_ratio, source=source,
                         interaction_model=interaction_model,
                         earth_model=earth_model)

    @property
    def volume(self):
        """
        Generation volume (m^3) in which event vertices are produced.

        """
        return np.pi * self.dr**2 * self.dz

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
        r = self.dr * np.sqrt(np.random.random_sample())
        theta = 2*np.pi * np.random.random_sample()
        z = -self.dz * np.random.random_sample()
        return np.array([r*np.cos(theta), r*np.sin(theta), z])

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

        # Find the intersection points of the circle, assuming infinite z
        if particle.direction[0]==0:
            x0 = particle.vertex[0]
            y0 = -np.sqrt(self.dr**2 - x0**2)
            z0 = (particle.vertex[2] + (y0-particle.vertex[1])
                  * particle.direction[2]/particle.direction[1])
            x1 = particle.vertex[0]
            y1 = np.sqrt(self.dr**2 - x1**2)
            z1 = (particle.vertex[2] + (y1-particle.vertex[1])
                  * particle.direction[2]/particle.direction[1])
        else:
            slope = particle.direction[1]/particle.direction[0]
            a = 1 + slope**2
            b = particle.vertex[1] - slope*particle.vertex[0]
            x0 = - (slope*b + np.sqrt(-b**2 + a*self.dr**2)) / a
            y0 = (particle.vertex[1] - slope *
                  (particle.vertex[0] + np.sqrt(-b**2 + a*self.dr**2))) / a
            z0 = (particle.vertex[2] + (x0-particle.vertex[0])
                  * particle.direction[2]/particle.direction[0])
            x1 = (-slope*b + np.sqrt(-b**2 + a*self.dr**2)) / a
            y1 = (particle.vertex[1] + slope *
                  (-particle.vertex[0] + np.sqrt(-b**2 + a*self.dr**2))) / a
            z1 = (particle.vertex[2] + (x1-particle.vertex[0])
                  * particle.direction[2]/particle.direction[0])

        for pt in ([x0, y0, z0], [x1, y1, z1]):
            # Check for intersections at the top & bottom that supersede the
            # intersections at the sides
            z = None
            if pt[2]>0:
                z = 0
            elif pt[2]<-self.dz:
                z = -self.dz
            if z is not None:
                pt[0] = (particle.vertex[0] + (z-particle.vertex[2])
                         * particle.direction[0]/particle.direction[2])
                pt[1] = (particle.vertex[1] + (z-particle.vertex[2])
                         * particle.direction[1]/particle.direction[2])
                pt[2] = z
            pt = np.array(pt)
            # Sort into enter and exit points based on particle direction
            nonzero = particle.direction!=0
            direction = ((pt[nonzero]-particle.vertex[nonzero])
                         /particle.direction[nonzero])
            if np.all(direction<0):
                enter_point = pt
            elif np.all(direction>0):
                exit_point = pt
            elif np.all(direction==0):
                if enter_point is None:
                    enter_point = pt
                if exit_point is None:
                    exit_point = pt

        if enter_point is not None and exit_point is not None:
            return enter_point, exit_point
        else:
            raise ValueError("Could not determine exit points")


class RectangularGenerator(Generator):
    """
    Class to generate neutrino vertices in a rectangular ice volume.

    Generates neutrinos in a box with given width, length, and height.

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
    shadow : bool, optional
        Whether Earth shadowing effects should be used to reject events. If
        ``True`` then neutrinos which don't survive transit through the Earth
        will be skipped when creating events. If ``False`` then all events are
        allowed and assigned a weight to scale their probability of occurrence.
    flavor_ratio : array_like, optional
        Flavor ratio of neutrinos to be generated. Of the form [electron, muon,
        tau] neutrino fractions.
    source : optional
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : optional
        Class to use to describe interactions of the generated particles.
        Should inherit from (or behave like) the base ``Interaction`` class.

    Attributes
    ----------
    count : int
        Number of neutrinos produced by the generator, including those not
        returned due to Earth shadowing or other effects.
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
    shadow : bool
        Whether Earth shadowing effects will be used to reject events.
    ratio : ndarray
        (Normalized) flavor ratio of neutrinos to be generated. Of the form
        [electron, muon, tau] neutrino fractions.
    source : Generator.SourceType
        Source type of neutrinos to be generated. Used in the determination of
        per-flavor neutrino/antineutrino fractions.
    interaction_model : Interaction
        Class to use to describe interactions of the generated particles.
    volume
    solid_angle

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.

    """
    def __init__(self, dx, dy, dz, energy, shadow=False, flavor_ratio=(1,1,1),
                 source="cosmogenic", interaction_model=NeutrinoInteraction,
                 earth_model=earth):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        super().__init__(energy=energy, shadow=shadow,
                         flavor_ratio=flavor_ratio, source=source,
                         interaction_model=interaction_model,
                         earth_model=earth_model)

    @property
    def volume(self):
        """
        Generation volume (m^3) in which event vertices are produced.

        """
        return self.dx * self.dy * self.dz

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
    count : int
        Number of neutrinos produced by the generator, including those not
        returned due to Earth shadowing or other effects.
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
        if (isinstance(events, Iterable) and
                not isinstance(events, Event)):
            self.events = events
        else:
            self.events = [events]
        for i, event in enumerate(self.events):
            if isinstance(event, Particle):
                self.events[i] = Event(event)
        self.loop = loop
        self._index = 0
        self._additional_counts = 0

    @property
    def count(self):
        """
        Number of neutrinos produced by the generator.

        Count includes events which were not returned due to Earth shadowing
        or other effects.

        """
        return self._index + self._additional_counts

    @count.setter
    def count(self, custom_count):
        self._additional_counts = custom_count - self._index

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
        if not self.loop and self._index>=len(self.events):
            raise StopIteration("No more events to be generated")
        self._index += 1
        return self.events[(self._index-1)%len(self.events)]


class FileGenerator:
    """
    Class to generate neutrino events from simulation file(s).

    Generates neutrinos by pulling their attributes from a (list of) simulation
    output file(s). Designed to make reproducing simulations easier.

    Parameters
    ----------
    files : str or list of str
        List of file names containing neutrino event information. If only a
        single file name is provided, creates a list with that file alone.
    slice_range : int, optional
        Number of events to load into memory at a time from the files.
        Increasing this value should result in an improvement in speed, while
        decreasing this value should result in an improvement in memory
        consumption.
    interaction_model : optional
        Class used to describe the interactions of the stored particles.

    Attributes
    ----------
    count : int
        Number of neutrinos produced by the generator, including those not
        returned due to Earth shadowing or other effects.
    files : list of str
        List of file names containing neutrino information.

    Warnings
    --------
    This generator only supports `Event` objects containing a single level of
    `Particle` objects. Any dependencies among `Particle` objects will be
    ignored and they will all appear in the root level.

    See Also
    --------
    pyrex.particle.Interaction : Base class for describing neutrino interaction
                                 attributes.
    pyrex.Event : Class for storing a tree of `Particle` objects
                  representing an event.
    pyrex.Particle : Class for storing particle attributes.

    """
    def __init__(self, files, slice_range=100,
                 interaction_model=NeutrinoInteraction):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.slice_range = slice_range
        self.interaction_model = interaction_model
        self._file_index = -1
        self._file_counts = [0] * (len(self.files)+1)
        self._load_events()

    @property
    def count(self):
        """
        Number of neutrinos produced by the generator.

        Count includes events which were not returned due to Earth shadowing
        or other effects.

        """
        return sum(self._file_counts)

    @count.setter
    def count(self, custom_count):
        self._file_counts[0] = custom_count - sum(self._file_counts[1:])

    def _load_events(self):
        """
        Pulls the next chunk of events into memory.

        Reads events up to the ``slice_range`` into memory from the current
        file. If the current file is exhausted, loads the next file.

        Returns
        -------
        list
            List of `Event` objects read from the current file.

        Raises
        ------
        StopIteration
            If the end of the last file in the file list has been reached.

        """
        if self._file_index<0 or self._event_index>=len(self._file):
            self._next_file()
        start = self._event_index
        stop = self._event_index + self.slice_range
        self._event_index += self.slice_range
        if stop>len(self._file):
            stop = len(self._file)
        self._events = []
        self._event_counts = []
        for file_event in self._file[start:stop]:
            info = file_event.get_particle_info()
            particles = []
            for p in info:
                part = Particle(
                    particle_id=p['particle_id'],
                    vertex=(p['vertex_x'],
                            p['vertex_y'],
                            p['vertex_z']),
                    direction=(p['direction_x'],
                               p['direction_y'],
                               p['direction_z']),
                    energy=p['energy'],
                    interaction_model=self.interaction_model,
                    interaction_type=p['interaction_kind']
                )
                part.interaction.inelasticity = p['interaction_inelasticity']
                part.interaction.em_frac = p['interaction_em_frac']
                part.interaction.had_frac = p['interaction_had_frac']
                part.survival_weight = p['survival_weight']
                part.interaction_weight = p['interaction_weight']
                particles.append(part)
            self._events.append(Event(particles))
            self._event_counts.append(file_event.total_events_thrown)

    def _next_file(self):
        """
        Pulls the next file into memory.

        Reads in the next file from the ``files`` list and stores its `Event`
        objects in memory.

        Raises
        ------
        StopIteration
            If the end of the last file in the file list has been reached.

        """
        self._file_index += 1
        self._event_index = 0
        if self._file_index>0:
            self._file.close()
        if self._file_index>=len(self.files):
            raise StopIteration("No more events to be generated")
        # Try to open the next file with the appropriate slice range,
        # otherwise just settle for opening it at all
        try:
            self._file = File(self.files[self._file_index], 'r',
                              slice_range=self.slice_range)
        except TypeError:
            self._file = File(self.files[self._file_index], 'r')
        self._file.open()

    def create_event(self):
        """
        Generate a neutrino.

        Pulls the next `Event` object from the file(s).

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
        if len(self._events)==0:
            self._load_events()
        self._file_counts[self._file_index+1] = self._event_counts.pop(0)
        return self._events.pop(0)
