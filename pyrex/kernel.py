"""
Module for the simulation kernel.

The simulation kernel is responsible for running through the simulation
chain by controlling classes and objects which will independently produce
neutrinos, create corresponding signals, propagate the signals to antennas,
and handle antenna processing of the signals.

"""

import logging
import numpy as np
from pyrex.internal_functions import normalize
from pyrex.signals import AskaryanSignal
from pyrex.ray_tracing import RayTracer
from pyrex.ice_model import IceModel

logger = logging.getLogger(__name__)


class EventKernel:
    """
    High-level kernel for controlling event simulation.

    The kernel is responsible for handling the classes and objects which
    control the major simulation steps: particle creation, signal production,
    signal propagation, and antenna response. The modular kernel structure
    allows for easy switching of the classes or objects which handle any of the
    simulation steps.

    Parameters
    ----------
    generator
        A particle generator to create neutrino events.
    antennas
        An iterable object consisting of antenna objects which can receive and
        store signals.
    ice_model : optional
        An ice model describing the ice surrounding the `antennas`.
    ray_tracer : optional
        A ray tracer capable of propagating signals from the neutrino vertex
        to the antenna positions.
    signal_model : optional
        A signal class which generates signals based on the particle.
    signal_times : array_like, optional
        The array of times over which the neutrino signal should be generated.

    Attributes
    ----------
    gen
        The particle generator responsible for particle creation.
    antennas
        The iterable of antennas responsible for handling applying their
        response and storing the resulting signals.
    ice
        The ice model describing the ice containing the `antennas`.
    ray_tracer
        The ray tracer responsible for signal propagation through the `ice`.
    signal_model
        The signal class to use to generate signals based on the particle.
    signal_times
        The array of times over which the neutrino signal should be generated.

    See Also
    --------
    pyrex.Event : Class for storing a tree of `Particle` objects
                  representing an event.
    pyrex.Particle : Class for storing particle attributes.
    pyrex.IceModel : Class describing the ice at the south pole.
    pyrex.RayTracer : Class for calculating the ray-trace solutions between
                      points.
    pyrex.AskaryanSignal : Class for generating Askaryan signals according to
                           ARVZ parameterization.

    Notes
    -----
    The kernel is designed to be modular so individual parts of the simulation
    chain can be exchanged. In order to interchange the pieces, their classes
    require the following at a minimum:

    The particle generator `generator` must have a ``create_event`` method
    which takes no arguments and returns a `Event` object consisting of
    `Particle` objects with ``vertex``, ``direction``, ``energy``, and
    ``weight`` attributes.

    The antenna iterable `antennas` must yield each antenna object once when
    iterating directly over `antennas`. Each antenna object must have a
    ``position`` attribute and a ``receive`` method which takes a signal object
    as its first argument, and ``ndarray`` objects as ``direction`` and
    ``polarization`` keyword arguments.

    The `ice_model` must have an ``index`` method returning the index of
    refraction given a (negative-valued) depth, and it must support anything
    required of it by the `ray_tracer`.

    The `ray_tracer` must be initialized with the particle vertex and an
    antenna position as its first two arguments, and the `ice_model` of the
    kernel as the ``ice_model`` keyword argument. The ray tracer must also have
    ``exists`` and ``solutions`` attributes, the first of which denotes whether
    any paths exist between the given points and the second of which is an
    iterable revelaing each path between the points. These paths must have
    ``emitted_direction``, ``received_direction``, and ``path_length``
    attributes, as well as a ``propagate`` method which takes a signal object
    and applies the propagation effects of the path in-place to that object.

    The `signal_model` must be initialized with the `signal_times` array,
    a `Particle` object from the `Event`, the ``viewing_angle`` and
    ``viewing_distance`` according to the `ray_tracer`, and the `ice_model`.
    The object created should be a `Signal` object with ``times`` and
    ``values`` attributes representing the time-domain Askaryan signal produced
    by the `Particle`.

    """
    def __init__(self, generator, antennas, ice_model=IceModel,
                 ray_tracer=RayTracer, signal_model=AskaryanSignal,
                 signal_times=np.linspace(-20e-9, 80e-9, 2000, endpoint=False)):
        self.gen = generator
        self.antennas = antennas
        self.ice = ice_model
        self.ray_tracer = ray_tracer
        self.signal_model = signal_model
        self.signal_times = signal_times

    def event(self):
        """
        Create a neutrino event and run it through the simulation chain.

        Creates a particle using the ``generator``, produces a signal from that
        event, propagates that signal through the ice according to the
        ``ice_model`` and the ``ray_tracer``, and passes it into the
        ``antennas`` for processing.

        Returns
        -------
        Event
            The neutrino event generated which is responsible for the waveforms
            on the antennas.

        See Also
        --------
        pyrex.Event : Class for storing a tree of `Particle` objects
                      representing an event.
        pyrex.Particle : Class for storing particle attributes.

        """
        event = self.gen.create_event()
        for particle in event:
            logger.info("Processing event for %s", particle)
            for ant in self.antennas:
                rt = self.ray_tracer(particle.vertex, ant.position,
                                     ice_model=self.ice)

                # If no path(s) between the points, skip ahead
                if not rt.exists:
                    logger.debug("Ray paths to %s do not exist", ant)
                    continue

                for path in rt.solutions:
                    # epol is (negative) vector rejection of
                    # path.received_direction onto particle.direction,
                    # making epol orthogonal to path.recieved_direction in the
                    # same plane as p.direction and path.received_direction
                    epol = normalize(np.vdot(path.received_direction,
                                             particle.direction)
                                     * path.received_direction
                                     - particle.direction)
                    # In case path.received_direction and particle.direction
                    # are equal, just let epol be all zeros

                    psi = np.arccos(np.vdot(particle.direction,
                                            path.emitted_direction))
                    logger.debug("Angle to %s is %f degrees", ant,
                                 np.degrees(psi))
                    # TODO: Support angles larger than pi/2
                    # (low priority since these angles are far from the
                    # cherenkov cone)
                    if psi>np.pi/2:
                        continue

                    pulse = self.signal_model(times=self.signal_times,
                                              particle=particle,
                                              viewing_angle=psi,
                                              viewing_distance=path.path_length,
                                              ice_model=self.ice)

                    path.propagate(pulse)

                    ant.receive(pulse, direction=path.received_direction,
                                polarization=epol)

        return event
