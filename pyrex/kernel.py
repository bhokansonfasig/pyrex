"""
Module for the simulation kernel.

The simulation kernel is responsible for running through the simulation
chain by controlling classes and objects which will independently produce
neutrinos, create corresponding signals, propagate the signals to antennas,
and handle antenna processing of the signals.

"""

from collections.abc import Sequence
import logging
import numpy as np
from pyrex.internal_functions import normalize
from pyrex.signals import EmptySignal
from pyrex.askaryan import AskaryanSignal
from pyrex.ray_tracing import RayTracer
from pyrex.ice_model import ice

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
    event_writer : File, optional
        A file object to be used for writing data output.
    triggers : function or dict, optional
        A function or dictionary with function values representing trigger
        conditions of the detector. If a dictionary, must have a "global" key
        with its value representing the global detector trigger.
    offcone_max : float or None, optional
        The maximum angle away from the Cherenkov angle to be simulated.
        Antennas which view an event with an angle larger than this angle will
        skip the calculation of the Askaryan signal and assume no significant
        signal is seen. If `None`, no offcone cut is applied.
    weight_min : float or tuple or None, optional
        The minimum particle weight(s) which should be simulated. If a float,
        particles with a total weight less than this value will be skipped. If
        a tuple, particles with a survival weight less than the first element
        of the tuple or with an interaction weight less than the second element
        of the tuple will be skipped. If `None`, no minimum weight is applied.
    attenuation_interpolation : float or None, optional
        The logarithmic (base 10) interpolation step size to be used for
        interpolating attenuation along the ray path. If `None`, no
        interpolation of the attenuation is applied.

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
    writer
        The file object to be used for writing data output.
    triggers
        The trigger condition(s) of the detector.
    offcone_max
        The maximum angle away from the Cherenkov angle to be simulated.
    weight_min
        The minimum particle weight(s) which should be simulated.
    attenuation_interpolation : float or None, optional
        The logarithmic (base 10) interpolation step size to be used for
        interpolating attenuation along the ray path.

    See Also
    --------
    pyrex.Event : Class for storing a tree of `Particle` objects
                  representing an event.
    pyrex.Particle : Class for storing particle attributes.
    pyrex.ice_model.AntarcticIce : Class describing the ice at the south pole.
    pyrex.RayTracer : Class for calculating the ray-trace solutions between
                      points.
    pyrex.AskaryanSignal : Class for generating Askaryan signals according to
                           ARZ parameterization.
    pyrex.File : Class for reading or writing data files.

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
    iterable revealing each path between the points. These paths must have
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
    def __init__(self, generator, antennas, ice_model=ice,
                 ray_tracer=RayTracer, signal_model=AskaryanSignal,
                 signal_times=np.linspace(-50e-9, 50e-9, 2000, endpoint=False),
                 event_writer=None, triggers=None, offcone_max=40,
                 weight_min=None, attenuation_interpolation=0.1):
        self.gen = generator
        self.antennas = antennas
        self.ice = ice_model
        self.ray_tracer = ray_tracer
        self.signal_model = signal_model
        self.signal_times = signal_times
        self.writer = event_writer
        self.triggers = triggers
        if offcone_max is None:
            self.offcone_max = np.radians(180)
        else:
            self.offcone_max = np.radians(offcone_max)
        if weight_min is None:
            self.weight_min = 0
        else:
            self.weight_min = weight_min
        self.attenuation_interpolation = attenuation_interpolation
        self._gen_count = self.gen.count
        if self.writer is not None:
            if not self.writer.is_open:
                logger.warning("Event writer was not open. Opening now.")
                self.writer.open()
            if not self.writer.has_detector:
                self.writer.set_detector(antennas)
            # Add metadata about the classes used
            kernel_metadata = {
                "detector_class": str(type(self.antennas)),
                "generator_class": str(type(self.gen)),
                "ice_model_class": str(type(self.ice)),
                "ray_tracer_class": str(self.ray_tracer),
                "signal_model_class": str(self.signal_model),
                "offcone_max": np.degrees(self.offcone_max),
                "attenuation_interpolation": (self.attenuation_interpolation
                                              if self.attenuation_interpolation
                                              is not None else 0),
            }
            if isinstance(self.weight_min, Sequence):
                kernel_metadata["survival_weight_min"] = self.weight_min[0]
                kernel_metadata["interaction_weight_min"] = self.weight_min[1]
            else:
                kernel_metadata["weight_min"] = self.weight_min
            try:
                kernel_metadata["earth_model_class"] = str(type(
                    self.gen.earth_model
                ))
            except AttributeError:
                pass
            self.writer.create_analysis_metadataset("sim_parameters")
            self.writer.add_analysis_metadata("sim_parameters", kernel_metadata)

    def event(self):
        """
        Create a neutrino event and run it through the simulation chain.

        Creates a particle using the ``generator``, produces a signal from that
        event, propagates that signal through the ice according to the
        ``ice_model`` and the ``ray_tracer``, and passes it into the
        ``antennas`` for processing.

        Returns
        -------
        event : Event
            The neutrino event generated which is responsible for the waveforms
            on the antennas.
        triggered : bool, optional
            If the ``triggers`` parameter was specified, contains whether the
            global trigger condition of the detector was met.

        See Also
        --------
        pyrex.Event : Class for storing a tree of `Particle` objects
                      representing an event.
        pyrex.Particle : Class for storing particle attributes.

        """
        event = self.gen.create_event()
        ray_paths = []
        polarizations = []
        for i in range(len(self.antennas)):
            ray_paths.append([])
            polarizations.append([])
        for particle in event:
            logger.info("Processing event for %s", particle)
            if isinstance(self.weight_min, Sequence):
                if ((particle.survival_weight is not None and
                        particle.survival_weight<self.weight_min[0]) or
                        (particle.interaction_weight is not None and
                         particle.interaction_weight<self.weight_min[1])):
                    logger.debug("Skipping particle with weight below %s",
                                 self.weight_min)
                    continue
            elif particle.weight<self.weight_min:
                logger.debug("Skipping particle with weight below %s",
                             self.weight_min)
                continue

            for i, ant in enumerate(self.antennas):
                rt = self.ray_tracer(particle.vertex, ant.position,
                                     ice_model=self.ice)

                # If no path(s) between the points, skip ahead
                if not rt.exists:
                    logger.debug("Ray paths to %s do not exist", ant)
                    continue

                theta_c = np.arccos(1/self.ice.index(particle.vertex[2]))

                ray_paths[i].extend(rt.solutions)
                for path in rt.solutions:
                    # nu_pol is the signal polarization at the neutrino vertex
                    # It's calculated as the (negative) vector rejection of
                    # path.emitted_direction onto particle.direction, making
                    # epol orthogonal to path.emitted_direction in the same
                    # plane as particle.direction and path.emitted_direction
                    # This is equivalent to the vector triple product
                    # (particle.direction x path.emitted_direction) x
                    # path.emitted_direction
                    # In the case when path.emitted_direction and
                    # particle.direction are equal, just let nu_pol be zeros
                    nu_pol = normalize(np.vdot(path.emitted_direction,
                                               particle.direction)
                                       * path.emitted_direction
                                       - particle.direction)
                    polarizations[i].append(nu_pol)

                    psi = np.arccos(np.vdot(particle.direction,
                                            path.emitted_direction))
                    logger.debug("Angle to %s is %f degrees", ant,
                                 np.degrees(psi))

                    try:
                        if np.abs(psi-theta_c)>self.offcone_max:
                            raise ValueError("Viewing angle is larger than "+
                                             "offcone limit "+
                                             str(np.degrees(self.offcone_max)))
                        pulse = self.signal_model(
                            times=self.signal_times,
                            particle=particle,
                            viewing_angle=psi,
                            viewing_distance=path.path_length,
                            ice_model=self.ice
                        )
                    except ValueError as err:
                        logger.debug("Eliminating invalid Askaryan signal: %s",
                                     err)
                        ant.receive(
                            EmptySignal(self.signal_times+path.tof,
                                        value_type=EmptySignal.Type.field)
                        )
                    else:
                        ant_pulses, ant_pols = path.propagate(
                            signal=pulse, polarization=nu_pol,
                            attenuation_interpolation=self.attenuation_interpolation
                        )
                        ant.receive(
                            ant_pulses,
                            direction=path.received_direction,
                            polarization=ant_pols
                        )

        if self.triggers is None:
            triggered = None
        elif isinstance(self.triggers, dict):
            triggered = {key: trigger_func(self.antennas)
                         for key, trigger_func in self.triggers.items()}
        else:
            triggered = self.triggers(self.antennas)

        if self.writer is not None:
            self.writer.add(event=event, triggered=triggered,
                            ray_paths=ray_paths, polarizations=polarizations,
                            events_thrown=self.gen.count-self._gen_count)

        self._gen_count = self.gen.count

        if triggered is None:
            return event
        elif isinstance(self.triggers, dict):
            return event, triggered['global']
        else:
            return event, triggered
