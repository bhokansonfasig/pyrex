"""
Module containing classes for ray tracing through layered ice.

Supports layers of ice where the index is monotonic within the layer's valid
range.

"""

import logging
import numpy as np
import scipy.optimize
from pyrex.internal_functions import (flatten, normalize,
                                      LazyMutableClass, lazy_property)
from pyrex.ice_model import AntarcticIce, UniformIce
from pyrex.ray_tracing import (BasicRayTracer, SpecializedRayTracer,
                               UniformRayTracer)
from .ice_model import LayeredIce

logger = logging.getLogger(__name__)


class LayeredRayTracePath(LazyMutableClass):
    """
    Class for representing a single ray solution in layered ice.

    Stores parameters of the ray path along the layers of ice traversed.
    Most methods dispatch to the matching methods of the layers and combine the
    results appropriately. Most properties are lazily evaluated to save on
    computation time. If any attributes of the class instance are changed, the
    lazily-evaluated properties will be cleared.

    Parameters
    ----------
    parent_tracer : LayeredRayTracer
        Ray tracer for which this path is a solution.
    paths : array_like
        Array of path objects corresponding to the ice layers in the order of
        traversal.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    emitted_direction
    received_direction
    path_length
    tof
    coordinates

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    LayeredRayTracer : Class for calculating the ray solutions in layered ice.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    def __init__(self, parent_tracer, paths):
        self.from_point = parent_tracer.from_point
        self.to_point = parent_tracer.to_point
        self.ice = parent_tracer.ice
        self.paths = paths
        super().__init__()

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used is supported."""
        valid = isinstance(self.ice, LayeredIce)
        if valid:
            for path in self.paths:
                valid = valid and path.valid_ice_model
        return valid

    @property
    def z0(self):
        """Depth (m) of the launching point."""
        return self.from_point[2]

    @property
    def z1(self):
        """Depth (m) of the receiving point."""
        return self.to_point[2]

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the launching point."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance (m) between the endpoints."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def phi(self):
        """Azimuthal angle (radians) between the endpoints."""
        u = self.to_point - self.from_point
        return np.arctan2(u[1], u[0])

    @lazy_property
    def emitted_direction(self):
        """Direction in which ray is emitted."""
        return self.paths[0].emitted_direction

    @lazy_property
    def received_direction(self):
        """Direction ray is travelling when it is received."""
        return self.paths[-1].received_direction

    @lazy_property
    def path_length(self):
        """Length (m) of the ray path."""
        return np.sum([path.path_length for path in self.paths])

    @lazy_property
    def tof(self):
        """Time of flight (s) along the ray path."""
        return np.sum([path.tof for path in self.paths])

    @lazy_property
    def fresnel(self):
        """
        Fresnel factors for reflection and transmission along the path.

        The fresnel reflectance and transmittance at each layer are calculated
        as the square root (ratio of amplitudes, not powers). Stores the s and
        p polarized factors, respectively.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of ice")
        f_s, f_p = self.paths[0].fresnel
        for path_1, path_2 in zip(self.paths[:-1], self.paths[1:]):
            n_1 = path_1.ice.index(path_1.to_point[2])
            theta_1 = np.arccos(path_1.received_direction[2])
            if np.sign(path_1.received_direction[2])<0:
                theta_1 = np.pi - theta_1
            cos_1 = np.cos(theta_1)

            if (np.sign(path_1.received_direction[2])
                    !=np.sign(path_2.emitted_direction[2])):
                # Reflection
                try:
                    i = self.ice.boundaries.index(path_1.to_point[2])
                except ValueError:
                    for i, bound in enumerate(self.ice.boundaries):
                        if np.isclose(path_1.to_point[2], bound, rtol=0):
                            break
                    else:
                        raise
                if path_1.received_direction[2]>0:
                    if i==0:
                        n_2 = self.ice.index_above
                    else:
                        n_2 = self.ice.layers[i-1].index(path_2.from_point[2])
                else:
                    if i==len(self.ice.layers):
                        n_2 = self.ice.index_below
                    else:
                        n_2 = self.ice.layers[i].index(path_2.from_point[2])
                sin_2 = n_1/n_2*np.sin(theta_1)
                if sin_2<=1:
                    cos_2 = np.sqrt(1 - (sin_2)**2)
                else:
                    cos_2 = np.sqrt((sin_2)**2 - 1)*1j
                # TODO: Confirm sign convention here
                r_s = (n_1*cos_1 - n_2*cos_2) / (n_1*cos_1 + n_2*cos_2)
                r_p = (n_2*cos_1 - n_1*cos_2) / (n_2*cos_1 + n_1*cos_2)
                f_s *= r_s
                f_p *= r_p
            else:
                # Transmission
                n_2 = path_2.ice.index(path_2.from_point[2])
                sin_2 = n_1/n_2*np.sin(theta_1)
                if sin_2<=1:
                    cos_2 = np.sqrt(1 - (sin_2)**2)
                else:
                    cos_2 = np.sqrt((sin_2)**2 - 1)*1j
                # TODO: Confirm sign convention here
                t_s = (2*n_1*cos_1) / (n_1*cos_1 + n_2*cos_2)
                t_p = (2*n_1*cos_1) / (n_2*cos_1 + n_1*cos_2)
                f_s *= t_s
                f_p *= t_p

            # Include any fresnel coefficients from along the path
            path_f_s, path_f_p = path_2.fresnel
            f_s *= path_f_s
            f_p *= path_f_p

        return f_s, f_p

    def attenuation(self, f, *args, **kwargs):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        *args, **kwargs
            Arguments passed down to individual path attenuation methods.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        try:
            attens = np.ones(f.shape)
        except AttributeError:
            attens = 1
        for path in self.paths:
            attens *= path.attenuation(f, *args, **kwargs)
        return attens

    def propagate(self, signal=None, polarization=None):
        """
        Propagate the signal with optional polarization along the ray path.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight. Additionally
        provides the s and p polarization directions.

        Parameters
        ----------
        signal : Signal, optional
            ``Signal`` object to propagate.
        polarization : array_like, optional
            Vector representing the linear polarization of the `signal`.

        Returns
        -------
        tuple of Signal
            Tuple of ``Signal`` objects representing the s and p polarizations
            of the original `signal` attenuated along the ray path. Only
            returned if `signal` was not ``None``.
        tuple of ndarray
            Tuple of polarization vectors representing the s and p polarization
            directions of the `signal` at the end of the ray path. Only
            returned if `polarization` was not ``None``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if polarization is None:
            if signal is None:
                return

            else:
                new_signal = signal.copy()
                new_signal.shift(self.tof)
                new_signal.filter_frequencies(self.attenuation)
                return new_signal

        else:
            # Unit vectors perpendicular and parallel to plane of incidence
            # at the launching point
            u_s0 = normalize(np.cross(self.emitted_direction, [0, 0, 1]))
            u_p0 = normalize(np.cross(u_s0, self.emitted_direction))
            # Unit vector parallel to plane of incidence at the receiving point
            # (perpendicular vector stays the same)
            u_p1 = normalize(np.cross(u_s0, self.received_direction))

            if signal is None:
                return (u_s0, u_p1)

            else:
                # Amplitudes of s and p components
                pol_s = np.dot(polarization, u_s0)
                pol_p = np.dot(polarization, u_p0)
                # Fresnel reflectances of s and p components
                f_s, f_p = self.fresnel
                # Apply fresnel s and p coefficients in addition to attenuation
                attenuation_s = lambda freqs: self.attenuation(freqs) * f_s
                attenuation_p = lambda freqs: self.attenuation(freqs) * f_p
                signal_s = signal * pol_s
                signal_p = signal * pol_p
                signal_s.shift(self.tof)
                signal_p.shift(self.tof)
                signal_s.filter_frequencies(attenuation_s, force_real=True)
                signal_p.filter_frequencies(attenuation_p, force_real=True)
                return (signal_s, signal_p), (u_s0, u_p1)

    @lazy_property
    def coordinates(self):
        """
        x, y, and z-coordinates along the path.

        Coordinates are only calculated at ice layer boundaries, as the path
        is assumed to be straight within an ice layer.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of ice")
        xs = []
        ys = []
        zs = []
        for path in self.paths:
            x, y, z = path.coordinates
            xs.extend(x)
            ys.extend(y)
            zs.extend(z)
        return xs, ys, zs



class LayeredRayTracer(LazyMutableClass):
    """
    Class for calculating the ray solutions in layered ice.

    Calculates paths among the ice layers up to a maximum number of allowed
    reflections. Most properties are lazily evaluated to save on computation
    time. If any attributes of the class instance are changed, the
    lazily-evaluated properties will be cleared.

    Parameters
    ----------
    from_point : array_like
        Vector starting point of the ray path.
    to_point : array_like
        Vector ending point of the ray path.
    ice_model
        The ice model used for the ray tracer.

    Attributes
    ----------
    from_point : ndarray
        The starting point of the ray path.
    to_point : ndarray
        The ending point of the ray path.
    ice
        The ice model used for the ray tracer.
    max_reflections : int
        The maximum number of reflections allowed in a solution path.
    solution_class
        Class to be used for each ray-trace solution path.
    exists
    solutions

    See Also
    --------
    pyrex.internal_functions.LazyMutableClass : Class with lazy properties
                                                which may depend on other class
                                                attributes.
    LayeredRayTracePath : Class for representing a single ray solution in
                          layered ice.

    Notes
    -----
    Even more attributes than those listed are available for the class, but
    are mainly for internal use. These attributes can be found by exploring
    the source code.

    """
    solution_class = LayeredRayTracePath
    max_reflections = 1
    solution_sorting = lambda self, path: path.tof
    _angle_checks = 91
    _angle_precision = 1e-12
    ray_tracer_map = {
        UniformIce: UniformRayTracer,
        AntarcticIce: SpecializedRayTracer,
        'default': BasicRayTracer
    }

    def __init__(self, from_point, to_point, ice_model):
        self.from_point = np.array(from_point)
        self.to_point = np.array(to_point)
        self.ice = ice_model
        super().__init__()

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used is supported."""
        valid = isinstance(self.ice, LayeredIce)
        if valid:
            for ice in self.ice.layers:
                if (not isinstance(ice, UniformIce) and
                        not isinstance(ice, AntarcticIce)):
                    return False
        return valid

    @property
    def z0(self):
        """Depth (m) of the launching point."""
        return self.from_point[2]

    @property
    def z1(self):
        """Depth (m) of the receiving point."""
        return self.to_point[2]

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the launching point."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance between the endpoints."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def phi(self):
        """Azimuthal angle (radians) between the endpoints."""
        u = self.to_point - self.from_point
        return np.arctan2(u[1], u[0])

    @lazy_property
    def exists(self):
        """Boolean of whether any paths exist between the endpoints."""
        return len(self.solutions)>0

    @classmethod
    def _build_path(cls, path, direction, reflections, max_level):
        """
        Recursively create an index path through adjacent levels.

        Given the current level and direction, branches off into transmitted
        and reflected paths off the next level. Won't travel beyond the 0th or
        `max_level` levels, and limits the number of reflections allowed.

        Parameters
        ----------
        path : list of int
            The current index path so far.
        direction : int
            The direction of travel. +1 for moving up in indices, -1 for moving
            down in indices.
        reflections : int
            Number of reflections remaining for the path.
        max_level : int
            Maximum allowed level.

        Returns
        -------
        paths : list
            Branched list of paths starting with the given `path`.

        """
        level = path[-1]
        if (level==0 and direction==-1) or (level==max_level and direction==1):
            if reflections==0:
                return [path]
            else:
                return cls._build_path(path+(level,), -direction, reflections-1, max_level)
        else:
            if reflections==0:
                return cls._build_path(path+(level+direction,), direction, reflections, max_level)
            else:
                return [cls._build_path(path+(level+direction,), direction, reflections, max_level),
                        cls._build_path(path+(level,), -direction, reflections-1, max_level)]

    @lazy_property
    def _potential_paths(self):
        """
        Allowed paths between the endpoints starting upwards or downwards.

        Generates lists of index paths from the starting layer to the ending
        layer incorporating up to the maximum number of reflections. Provides
        sets of the paths that start by going upwards and by going downwards,
        respectively.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of ice")
        start = self.ice.layers.index(self.ice.layer_at_depth(self.z0))
        end = self.ice.layers.index(self.ice.layer_at_depth(self.z1))
        upward_paths = []
        upward_tree = self._build_path(path=(start,), direction=-1,
                                       reflections=self.max_reflections,
                                       max_level=len(self.ice.layers)-1)
        for path in flatten(upward_tree, dont_flatten=(tuple,)):
            endpoints = [i for i, level in enumerate(path) if level==end]
            subpaths = [path[:i+1] for i in endpoints]
            upward_paths.extend(subpaths)
        downward_paths = []
        downward_tree = self._build_path(path=(start,), direction=1,
                                         reflections=self.max_reflections,
                                         max_level=len(self.ice.layers)-1)
        for path in flatten(downward_tree, dont_flatten=(tuple,)):
            endpoints = [i for i, level in enumerate(path) if level==end]
            subpaths = [path[:i+1] for i in endpoints]
            downward_paths.extend(subpaths)
        return set(upward_paths), set(downward_paths)


    def _get_matching_ray_tracer(self, ice_model):
        """
        Gets the ray tracer class corresponding to the given ice model.

        Paramters
        ---------
        ice_model
            Ice model object which has a matching ray tracer.

        Returns
        -------
        ray_tracer
            Ray tracer class from `ray_tracer_map` that matches the given
            `ice_model`.

        Raises
        ------
        ValueError
            If the given `ice_model` doesn't have a matching ray tracer in the
            `ray_tracer_map`.

        """
        # Check exact class matches first
        for key, val in self.ray_tracer_map.items():
            if key=='default':
                continue
            if type(ice_model)==key:
                return val
        # Then check for subclass matches too
        for key, val in self.ray_tracer_map.items():
            if key=='default':
                continue
            if isinstance(ice_model, key):
                return val
        if 'default' in self.ray_tracer_map:
            return self.ray_tracer_map['default']
        else:
            raise ValueError("No matching ray tracer for ice model "
                             +str(ice_model))

    def _build_path_at_layer(self, ice_layer, from_point, to_point, theta0, direct):
        """
        Creates a path object for given ice layer with starting parameters.

        Parameters
        ----------
        ice_layer
            Ice model object for a single layer of the ice.
        from_point : array_like
            Starting point of the path in the ice layer.
        to_point : array_like
            Ending point of the path in the ice layer.
        theta0 : float
            Launch angle of the path at `from_point`.
        direct : bool
            Whether the path goes directly between the points (`False` if a
            reflection or other directional inversion is made along the path).

        Returns
        -------
        path
            Path object representing the path in the given `ice_layer`.

        Raises
        ------
        ValueError
            If the corresponding ray tracer for `ice_layer` is not supported.

        See Also
        --------
        LayeredRayTracer._get_matching_ray_tracer : Gets the ray tracer class
                                                    corresponding to the given
                                                    ice model.

        """
        tracer_class = self._get_matching_ray_tracer(ice_layer)
        path_class = tracer_class.solution_class
        if issubclass(tracer_class, UniformRayTracer):
            tracer = tracer_class(from_point=from_point, to_point=to_point,
                                  ice_model=ice_layer)
            tracer.max_reflections = 0 if direct else 1
            return path_class(tracer, theta0, 0 if direct else 1)
        elif issubclass(tracer_class, (SpecializedRayTracer, BasicRayTracer)):
            tracer = tracer_class(from_point=from_point, to_point=to_point,
                                  ice_model=ice_layer)
            return path_class(tracer, theta0, direct)
        else:
            raise ValueError("Ice model "+str(ice_layer)+" not supported")


    def _get_radial_distance(self, angle, ice_layer, zs):
        """
        Calculates the radial distance traveled in an ice layer.

        Parameters
        ----------
        angle : float
            Launch angle of the ray as it enters the `ice_layer`.
        ice_layer
            Ice model object for a single layer of the ice.
        zs : array_like
            Array of depths for endpoints in the `ice_layer` (length of 2 for
            direct path, length of 3 for path with a single reflection).

        Returns
        -------
        distance : float
            Radial distance traveled by the path in `ice_layer` launched with
            `angle` and reaching the depths given by `zs`.

        Raises
        ------
        ValueError
            If the corresponding ray tracer for `ice_layer` is not supported.

        See Also
        --------
        LayeredRayTracer._get_matching_ray_tracer : Gets the ray tracer class
                                                    corresponding to the given
                                                    ice model.

        """
        tracer_class = self._get_matching_ray_tracer(ice_layer)
        if issubclass(tracer_class, UniformRayTracer):
            if angle==np.pi/2:
                if np.all(np.diff(zs)==0):
                    return None
                else:
                    return np.nan
            else:
                return np.sum(np.tan(angle) * np.diff(zs))
        elif issubclass(tracer_class, (SpecializedRayTracer, BasicRayTracer)):
            tracer = tracer_class(from_point=(0, 0, zs[0]),
                                  to_point=(0, 0, zs[-1]),
                                  ice_model=ice_layer)
            if zs[0]>zs[-1]:
                if len(zs)==2 and angle<np.pi/2:
                    return np.nan
                angle = np.arcsin(np.sin(angle)
                                  * ice_layer.index(zs[0])
                                  / ice_layer.index(zs[-1]))
            if len(zs)==2:
                if angle>tracer.max_angle:
                    return np.nan
                else:
                    return tracer._direct_r(angle)
            else:
                if angle>tracer.max_angle:
                    return np.nan
                else:
                    return tracer._indirect_r(angle)
        else:
            raise ValueError("Ice model "+str(ice_layer)+" not supported")

    def _check_reflection_depth(self, angle, ice_layer, zs):
        """
        Calculates the depth at which a reflection takes place in an ice layer.

        Parameters
        ----------
        angle : float
            Launch angle of the ray as it enters the `ice_layer`.
        ice_layer
            Ice model object for a single layer of the ice.
        zs : array_like
            Array of depths for endpoints in the `ice_layer` (length of 2 for
            direct path, length of 3 for path with a single reflection).

        Returns
        -------
        depth : float
            (Negative-valued) depth at which the reflection occurs in
            `ice_layer`. If the depth is not in `zs`, the reflection occurs
            within the layer rather than at an ice layer boundary.

        Raises
        ------
        ValueError
            If there was no reflection in the ice layer or if the corresponding
            ray tracer for `ice_layer` is not supported.

        See Also
        --------
        LayeredRayTracer._get_matching_ray_tracer : Gets the ray tracer class
                                                    corresponding to the given
                                                    ice model.

        """
        tracer_class = self._get_matching_ray_tracer(ice_layer)
        if len(zs)!=3:
            raise ValueError("Invalid number of path portions for reflection")
        if issubclass(tracer_class, UniformRayTracer):
            return zs[1]
        elif issubclass(tracer_class, (SpecializedRayTracer, BasicRayTracer)):
            tracer = tracer_class(from_point=(0, 0, zs[0]),
                                  to_point=(0, 0, zs[-1]),
                                  ice_model=ice_layer)
            return ice_layer.depth_with_index(tracer.n0 * np.sin(tracer.max_angle))
        else:
            raise ValueError("Ice model "+str(ice_layer)+" not supported")

    def _trace_path(self, angle, depths, grouped_path, models):
        """
        Traces out a ray path through the ice with a given launch `angle`.

        Parameters
        ----------
        angle : float
            Launch angle of the ray at it's starting point.
        depths : array_like
            Depths of ice layer boundaries along the expected path.
        grouped_path : array_like
            Array of indices for ice layers through which the path will pass.
        models : array_like
            Array of ice models for the ice layers of `grouped_path`.

        Returns
        -------
        radial_dists : array_like
            Array of radial distances traveled at each layer of the path.
        angles : array_like
            Array of launch angles at the starting point of each layer.

        """
        radial_dists = []
        angles = []
        start = 0
        for i, group in enumerate(grouped_path):
            stop = start + len(group)
            r = self._get_radial_distance(
                angle=angle, ice_layer=models[i],
                zs=depths[start:stop+1]
            )
            if r is None:
                r = self.rho - np.sum(radial_dists)
            radial_dists.append(r)
            angles.append(angle)
            if len(group)==2:
                # Reflection within layer -> final angle flipped
                angle = np.pi - angle
                # Check if reflection is at a boundary with an invalid index
                if ((group[-1]==0 and angle>np.pi/2
                     and self.ice._index_above is None) or
                        (group[-1]==len(self.ice.layers)-1 and angle<np.pi/2
                         and self.ice._index_below is None)):
                    z_turn = self._check_reflection_depth(
                        angle=angle, ice_layer=models[i],
                        zs=depths[start:stop+1]
                    )
                    if z_turn==depths[start+1]:
                        nans = [np.nan]*(len(grouped_path)-i-1)
                        radial_dists.extend(nans)
                        angles.extend(nans)
                        break
            if i<len(grouped_path)-1 and group[-1]!=grouped_path[i+1][0]:
                # Transmission -> angle transformed by Snell's law
                sin_angle = (np.sin(angle) * models[i].index(depths[start])
                             / models[i+1].index(depths[stop]))
                # Check if transmission is invalid due to total internal reflection
                if sin_angle>1:
                    nans = [np.nan]*(len(grouped_path)-i-1)
                    radial_dists.extend(nans)
                    angles.extend(nans)
                    break
                if angle<np.pi/2:
                    angle = np.arcsin(sin_angle)
                else:
                    angle = np.pi - np.arcsin(sin_angle)
            else:
                # Reflection -> angle flipped
                angle = np.pi - angle
                # Check if reflection is at a boundary with an invalid index
                if ((group[-1]==0 and angle>np.pi/2
                     and self.ice._index_above is None) or
                        (group[-1]==len(self.ice.layers)-1 and angle<np.pi/2
                         and self.ice._index_below is None)):
                    nans = [np.nan]*(len(grouped_path)-i-1)
                    radial_dists.extend(nans)
                    angles.extend(nans)
                    break
            start = stop
        return radial_dists, angles


    @lazy_property
    def solutions(self):
        """
        List of existing paths between the two points.

        Provides all possible paths between the points with up to the maximum
        number of reflections among the layers.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of ice")
        if not self.ice.contains(self.from_point):
            return []
        if not self.ice.contains(self.to_point):
            return []

        valid_paths = []
        used_paths = []
        for i, group in enumerate(self._potential_paths):
            start_direction = (-1)**(i+1)
            for path in group:
                # Skip potential duplicate direct path
                if (len(path)==1 and
                        ((self.z1-self.z0>=0 and start_direction==1) or
                         (self.z1-self.z0<0 and start_direction==-1))):
                    continue
                # Quick evaluation for direct path when endpoints are identical
                if len(path)==1 and np.array_equal(self.from_point, self.to_point):
                    sub_paths = [
                        self._build_path_at_layer(
                            self.ice.layers[path[0]], self.from_point,
                            self.to_point, theta0=0, direct=True)
                    ]
                    ray_path = self.solution_class(self, sub_paths)
                    valid_paths.append(ray_path)
                    used_paths.append((path, [start_direction]))
                    continue
                path_zs = [self.z0]
                directions = []
                direction = start_direction
                for level, next_level in zip(path[:-1], path[1:]):
                    path_zs.append(self.ice.boundaries[level+(direction+1)//2])
                    directions.append(direction)
                    if level==next_level:
                        direction *= -1
                directions.append(direction)
                path_zs.append(self.z1)
                path_zs = np.array(path_zs)
                directions = np.array(directions)
                # Skip duplicate points on boundaries at start and end
                if len(path_zs)>2 and path_zs[0]==path_zs[1]:
                    path = path[1:]
                    path_zs = path_zs[1:]
                    directions = directions[1:]
                if len(path_zs)>2 and path_zs[-1]==path_zs[-2]:
                    path = path[:-1]
                    path_zs = path_zs[:-1]
                    directions = directions[:-1]
                # Skip duplicate paths
                if (path, list(directions)) in used_paths:
                    continue

                ice_models = [self.ice.layers[j] for j in path]
                path_ns = np.array([ice.index(z) for ice, z in zip(ice_models, path_zs)])
                # Divide path into groups of one or two path sections based on
                # whether reflection can potentially be handled by a decreasing
                # index of refraction or not
                grouped_path = []
                group_models = []
                k = 0
                while k<len(path)-1:
                    if path[k]==path[k+1] and path_ns[k]>path_ns[k+1]:
                        grouped_path.append([path[k], path[k+1]])
                        group_models.append(ice_models[k])
                        k += 2
                    else:
                        grouped_path.append([path[k]])
                        group_models.append(ice_models[k])
                        k += 1
                if k<len(path):
                    grouped_path.append([path[k]])
                    group_models.append(ice_models[k])

                def distance(angle, tolerance=1e-12):
                    r = np.sum(self._trace_path(angle, path_zs, grouped_path,
                                                group_models)[0])
                    if np.abs(r)<tolerance:
                        return 0
                    else:
                        return r
                if start_direction==-1:
                    test_angles = np.linspace(0, np.pi/2,
                                              self._angle_checks)
                elif start_direction==1:
                    test_angles = np.linspace(np.pi/2, np.pi,
                                              self._angle_checks)
                else:
                    raise ValueError("Bad direction")
                test_radii = [distance(angle) for angle in test_angles]

                test_diffs = np.array(test_radii) - self.rho
                # Get indices of zeros
                angle_exact_idxs = np.where(test_diffs==0)[0]
                # Get indices around sign changes
                angle_min_idxs = np.where(
                    (np.diff(np.sign(test_diffs))!=0)
                    & (test_diffs[:-1]!=0) & (test_diffs[:-1]!=0)
                    & ~(np.isnan(test_diffs[:-1]) & np.isnan(test_diffs[1:]))
                )[0]
                angle_mins = []
                angle_maxs = []
                for idx in angle_exact_idxs:
                    angle_mins.append(test_angles[idx])
                    angle_maxs.append(test_angles[idx])
                for j, min_idx in enumerate(angle_min_idxs):
                    max_idx = min_idx+1
                    min_angle = test_angles[min_idx]
                    max_angle = test_angles[max_idx]
                    # Bring in min or max angle to non-nan values
                    if np.isnan(test_diffs[min_idx]):
                        min_angle = scipy.optimize.brentq(
                            lambda angle: int(np.isnan(distance(angle)))*2-1,
                            min_angle, max_angle, xtol=self._angle_precision,
                        )
                        if np.isnan(distance(min_angle)):
                            min_angle += self._angle_precision
                    if np.isnan(test_diffs[max_idx]):
                        max_angle = scipy.optimize.brentq(
                            lambda angle: int(np.isnan(distance(angle)))*2-1,
                            min_angle, max_angle, xtol=self._angle_precision,
                        )
                        if np.isnan(distance(max_angle)):
                            max_angle -= self._angle_precision
                    angle_mins.append(min_angle)
                    angle_maxs.append(max_angle)

                launch_angles = []
                for min_angle, max_angle in zip(angle_mins, angle_maxs):
                    try:
                        launch_angle = scipy.optimize.brentq(
                            lambda angle, true_dist: distance(angle) - true_dist,
                            min_angle, max_angle, args=(self.rho,),
                            xtol=self._angle_precision,
                        )
                    except ValueError:
                        # Bounds do not cross zero point -> not a valid pair
                        continue
                    # Skip identical paths from minimizer
                    if launch_angle in launch_angles:
                        continue
                    drs, angles = self._trace_path(launch_angle, path_zs,
                                                   grouped_path, group_models)
                    rs = np.cumsum(drs)
                    idxs = np.cumsum([len(elements) for elements in grouped_path], dtype=np.int_)
                    points = np.zeros((len(grouped_path)+1, 3))
                    points[0] = self.from_point
                    points[1:, 0] = self.from_point[0] + rs * np.cos(self.phi)
                    points[1:, 1] = self.from_point[1] + rs * np.sin(self.phi)
                    points[1:, 2] = path_zs[idxs]
                    points[-1] = self.to_point
                    sub_paths = [
                        self._build_path_at_layer(ice, p1, p2, angle, direct)
                        for ice, p1, p2, angle, direct in zip(
                            group_models, points[:-1], points[1:], angles,
                            [len(elements)==1 for elements in grouped_path]
                        )
                    ]
                    ray_path = self.solution_class(self, sub_paths)
                    valid_paths.append(ray_path)
                    used_paths.append((path, list(directions)))
                    launch_angles.append(launch_angle)

        if self.solution_sorting is None:
            return valid_paths
        else:
            return sorted(valid_paths, key=self.solution_sorting)
