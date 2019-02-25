"""
Module containing classes for ray tracing through layered ice.

Currently only ice layers with uniform indices of refraction are supported.
Theoretically it seems possible to eventually support layers of ice where the
index is monotonic.

"""

import collections
import logging
import numpy as np
import scipy.optimize
from pyrex.internal_functions import (flatten, normalize,
                                      LazyMutableClass, lazy_property)
from pyrex.signals import Signal
from .ice_model import UniformIce, LayeredIce

logger = logging.getLogger(__name__)


class LayeredRayTracePath(LazyMutableClass):
    """
    Class for representing a single ray solution in layered ice.

    Stores parameters of the ray path along the layers of ice traversed.
    Most properties are lazily evaluated to save on computation time. If any
    attributes of the class instance are changed, the lazily-evaluated
    properties will be cleared.

    Parameters
    ----------
    parent_tracer : LayeredRayTracer
        Ray tracer for which this path is a solution.
    path : array_like
        Array of indices corresponding to the ice layers in the order of
        traversal.
    initial_direction : int
        Initial direction the ray is traveling in the first ice layer. -1
        indicates the ray travels upwards (down in index values), +1 indicates
        the ray travels downwards (up in index values).
    drs : array_like
        Radial distances traversed by the ray in each layer in the path.

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
    def __init__(self, parent_tracer, path, initial_direction, drs):
        self.from_point = parent_tracer.from_point
        self.to_point = parent_tracer.to_point
        self.ice = parent_tracer.ice
        self._path = path
        self._drs = drs
        u = self.to_point - self.from_point
        u[2] = 0
        u = normalize(u)
        self._points = [self.from_point]
        self._directions = []
        direction = initial_direction
        for i, dr in enumerate(self._drs[:-1]):
            point = self._points[-1] + u*dr
            point[2] = self.ice.boundaries[path[i] + (direction+1)//2]
            self._points.append(point)
            self._directions.append(direction)
            if path[i+1]==path[i]:
                direction *= -1
        self._points.append(self.to_point)
        super().__init__()

    @lazy_property
    def valid_ice_model(self):
        """Whether the ice model being used is supported."""
        valid = isinstance(self.ice, LayeredIce)
        if valid:
            for ice in self.ice.layers:
                valid = valid and isinstance(ice, UniformIce)
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
        if (np.all(np.isclose(self._points[1], self._points[0]))
                and len(self._points)>2):
            return normalize(self._points[2] - self._points[1])
        else:
            return normalize(self._points[1] - self._points[0])

    @lazy_property
    def received_direction(self):
        """Direction ray is travelling when it is received."""
        if (np.all(np.isclose(self._points[-1], self._points[-2]))
                and len(self._points)>2):
            return normalize(self._points[-2] - self._points[-3])
        else:
            return normalize(self._points[-1] - self._points[-2])

    @lazy_property
    def path_length(self):
        """Length (m) of the ray path."""
        return np.sum([np.sqrt(np.sum((p2-p1)**2))
                       for p1, p2 in zip(self._points[:-1], self._points[1:])])

    @lazy_property
    def tof(self):
        """Time of flight (s) along the ray path."""
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of uniform ice")
        time = 0
        for p1, p2 in zip(self._points[:-1], self._points[1:]):
            z = (p1[2] + p2[2])/2
            index = self.ice.index(z)
            dist = np.sqrt(np.sum((p2-p1)**2))
            time += index * dist / 3e8
        return time

    @lazy_property
    def fresnel(self):
        """
        Fresnel factors for reflection off the ice surface.

        The fresnel reflectance and transmittance at each layer are calculated
        as the square root (ratio of amplitudes, not powers). Stores the s and
        p polarized factors, respectively.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of uniform ice")
        zs = self.ice.boundaries
        dzs = np.abs(np.diff(zs))
        indices = self.ice.index(zs)
        indices = np.concatenate(([self.ice.index_above], indices))

        start_dirn_idx = 1 - (self._directions[0]+1)//2
        end_dirn_idx = 1 - (self._directions[-1]+1)//2
        if len(self._path)==1:
            path_dzs = np.array([np.abs(self.from_point[2]-self.to_point[2])], dtype=np.float_)
            path_dzs[path_dzs==0] = 1e-20
        else:
            path_dzs = np.array([np.abs(self.from_point[2]-zs[self._path[0]+1-start_dirn_idx])]
                                + [dzs[i] for i in self._path[1:-1]]
                                + [np.abs(self.to_point[2]-zs[self._path[-1]+end_dirn_idx])],
                                dtype=np.float_)
            path_dzs[path_dzs==0] = 1e-20
        path_ns = np.array([indices[i+1] for i in self._path])

        f_s = 1
        f_p = 1
        for i in range(len(self._path)-1):
            dr = self._drs[i]
            dz = path_dzs[i]
            theta_1 = np.arctan(dr/dz)
            cos_1 = np.cos(theta_1)

            if self._path[i]==self._path[i+1]:
                # Reflection
                n_1 = path_ns[i]
                n_2 = indices[self._path[i] + 1 + self._directions[i]]
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
                n_1 = path_ns[i]
                n_2 = path_ns[i+1]
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

        return f_s, f_p

    def attenuation(self, f, dz=1):
        """
        Calculate the attenuation factor for signal frequencies.

        Calculates the attenuation factor to be multiplied by the signal
        amplitude at the given frequencies.

        Parameters
        ----------
        f : array_like
            Frequencies (Hz) at which to calculate signal attenuation.
        dz : float
            Step size to take in each layer of the ice. Actual step size will
            not be exactly this value, but is guaranteed to be less than
            the given value.

        Returns
        -------
        array_like
            Attenuation factors for the signal at the frequencies `f`.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of uniform ice")
        fa = np.abs(f)
        attens = np.ones(fa.shape)
        for p1, p2 in zip(self._points[:-1], self._points[1:]):
            if p1[2]==p2[2]:
                z = p1[2]
                zs = np.array([z])
                dp = np.sqrt(np.sum((p2-p1)**2))
            else:
                dpdz = (p2-p1)/(p2[2]-p1[2])
                z = (p1[2] + p2[2])/2
                n_steps = int(np.abs(p2[2]-p1[2]) / dz) + 2
                zs, dz_true = np.linspace(p1[2], p2[2], n_steps,
                                        endpoint=False, retstep=True)
                dp = np.sqrt(np.sum((dpdz*dz_true)**2))
            alens = self.ice.layer_at_depth(z).attenuation_length(zs, fa)
            attens *= np.prod(np.exp(-dp/alens), axis=0)
        return attens

    def propagate(self, signal=None, polarization=None):
        """
        Propagate the signal with optional polarization along the ray path.

        Applies the frequency-dependent signal attenuation along the ray path
        and shifts the times according to the ray time of flight. Additionally
        rotates the polarization vector appropriately.

        Parameters
        ----------
        signal : Signal, optional
            ``Signal`` object to propagate.
        polarization : array_like, optional
            Vector representing the linear polarization of the `signal`.

        Returns
        -------
        Signal
            ``Signal`` object representing the original `signal` attenuated
            along the ray path. Only returned if `signal` was not ``None``.
        array_like
            Polarization of the `signal` at the end of the ray path. Only
            returned if `polarization` was not ``None``.

        See Also
        --------
        pyrex.Signal : Base class for time-domain signals.

        """
        if signal is None and polarization is None:
            return

        if signal is not None:
            copy = Signal(signal.times+self.tof, signal.values,
                          value_type=signal.value_type)

        if polarization is None:
            copy.filter_frequencies(self.attenuation)
            return copy

        else:
            # Unit vectors perpendicular and parallel to plane of incidence
            # at the launching point
            u_s0 = normalize(np.cross(self.emitted_direction, [0, 0, 1]))
            u_p0 = normalize(np.cross(u_s0, self.emitted_direction))
            # Unit vector parallel to plane of incidence at the receiving point
            # (perpendicular vector stays the same)
            u_p1 = normalize(np.cross(u_s0, self.received_direction))
            # Amplitudes of s and p components
            pol_s = np.dot(polarization, u_s0)
            pol_p = np.dot(polarization, u_p0)
            # Fresnel reflectances of s and p components
            f_s, f_p = self.fresnel

            # Polarization vector at the receiving point
            receiving_polarization = normalize(pol_s*np.abs(f_s) * u_s0 +
                                               pol_p*np.abs(f_p) * u_p1)

            if signal is None:
                return receiving_polarization

            else:
                # Apply fresnel s and p coefficients in addition to attenuation
                # (In order to treat the phase delays of total internal
                # reflection properly, likely need a more robust framework
                # capable of handling elliptically polarized signals)
                def attenuation_with_fresnel(freqs):
                    return (self.attenuation(freqs) *
                            np.abs(np.sqrt(((f_s*pol_s)**2 + (f_p*pol_p)**2))))
                copy.filter_frequencies(attenuation_with_fresnel)

                return copy, receiving_polarization

    @lazy_property
    def coordinates(self):
        """
        x, y, and z-coordinates along the path.

        Coordinates are only calculated at ice layer boundaries, as the path
        is assumed to be straight within an ice layer.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of uniform ice")
        xs = np.array([p[0] for p in self._points])
        ys = np.array([p[1] for p in self._points])
        zs = np.array([p[2] for p in self._points])
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
                valid = valid and isinstance(ice, UniformIce)
        return valid

    # Calculations performed as if launching from low to high
    @property
    def z0(self):
        """
        Depth (m) of the lower endpoint.

        Ray tracing performed as if launching from lower point to higher point,
        since the only difference in the paths produced is a time reversal.
        This is the depth of the assumed launching point.

        """
        return min([self.from_point[2], self.to_point[2]])

    @property
    def z1(self):
        """
        Depth (m) of the higher endpoint.

        Ray tracing performed as if launching from lower point to higher point,
        since the only difference in the paths produced is a time reversal.
        This is the depth of the assumed receiving point.

        """
        return max([self.from_point[2], self.to_point[2]])

    @lazy_property
    def n0(self):
        """Index of refraction of the ice at the lower endpoint."""
        return self.ice.index(self.z0)

    @lazy_property
    def rho(self):
        """Radial distance between the endpoints."""
        u = self.to_point - self.from_point
        return np.sqrt(u[0]**2 + u[1]**2)

    @lazy_property
    def exists(self):
        """Boolean of whether any paths exist between the endpoints."""
        return len(self._potential_paths[0])+len(self._potential_paths[1])>0

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
            The direciton of travel. +1 for moving up in indices, -1 for moving
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
            raise TypeError("Ice model must consist of layers of uniform ice")
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

    @staticmethod
    def _path_equations(drs, dzs, indices, radius):
        """
        System of equations to be solved to determine radial distances.

        System of equations consists of the sum of `drs` being equal to the
        total radial distance `radius` and N-1 Snell's law equations between
        the starting layer and the current layer.

        """
        snells = indices * drs / np.sqrt(drs**2 + dzs**2)
        snell_diffs = snells-snells[0]
        return np.concatenate(([np.sum(drs)-radius], snell_diffs[1:]))

    @staticmethod
    def _path_equations_jacobian(drs, dzs, indices, radius):
        """
        Jacobian of the system of equations being solved for radial distances.

        System of equations consists of the sum of `drs` being equal to the
        total radial distance `radius` and N-1 Snell's law equations between
        the starting layer and the current layer.

        """
        derivs = indices**2 * dzs**2 / (drs**2 + dzs**2)**(3/2)
        jac = np.diag(derivs)
        jac[0] = -derivs[0]
        jac[:, 0] = 1
        return jac

    @lazy_property
    def solutions(self):
        """
        List of existing paths between the two points.

        Provides all possible paths between the points with up to the maximum
        number of reflections among the layers.

        """
        if not self.valid_ice_model:
            raise TypeError("Ice model must consist of layers of uniform ice")
        zs = self.ice.boundaries
        dzs = np.abs(np.diff(zs))
        indices = self.ice.index(zs)

        valid_paths = []
        for i, group in enumerate(self._potential_paths):
            start_direction = (-1)**(i+1)
            start_dirn_idx = 1 - (start_direction+1)//2
            for path in group:
                end_direction = (start_direction *
                                 (-1)**len(np.where(np.diff(path)==0)[0]))
                end_dirn_idx = 1 - (end_direction+1)//2
                if len(path)==1:
                    if ((self.z1-self.z0>=0 and start_direction==1) or
                            (self.z1-self.z0<0 and start_direction==-1)):
                        continue
                    path_dzs = np.array([np.abs(self.z0-self.z1)],
                                        dtype=np.float_)
                    path_dzs[path_dzs==0] = 1e-20
                    guess_drs = self.rho
                else:
                    path_dzs = np.array(
                        [np.abs(self.z0-zs[path[0]+1-start_dirn_idx])]
                        + [dzs[i] for i in path[1:-1]]
                        + [np.abs(self.z1-zs[path[-1]+end_dirn_idx])],
                        dtype=np.float_
                    )
                    path_dzs[path_dzs==0] = 1e-20
                    guess_drs = self.rho * path_dzs / np.sum(path_dzs)
                path_ns = np.array([indices[i] for i in path])
                sol = scipy.optimize.root(self._path_equations, guess_drs,
                                          args=(path_dzs, path_ns, self.rho),
                                          jac=self._path_equations_jacobian,
                                          options={'col_deriv': True})
                if sol.success:
                    if self.from_point[2]<=self.to_point[2]:
                        path = self.solution_class(
                            self, path, start_direction, sol.x
                        )
                    else:
                        path = self.solution_class(
                            self, list(reversed(path)), -1*end_direction,
                            list(reversed(sol.x))
                        )
                    # TODO: Eliminate paths based on attenuation
                    valid_paths.append(path)

        # TODO: Sort by tof
        return valid_paths
