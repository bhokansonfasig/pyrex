"""
Module containing classes for digital signal processing.

All classes in this module hold time-domain information about some signals,
and have methods for manipulating this data as it relates to digital signal
processing and general physics.

"""

from enum import Enum
import logging
import numpy as np
import scipy.signal
import scipy.fftpack
from pyrex.internal_functions import get_from_enum
from pyrex.ice_model import IceModel

logger = logging.getLogger(__name__)


class Signal:
    """
    Base class for time-domain signals.

    Stores the time-domain information for signal values. Supports adding
    between signals with the same times array and value type.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    values : array_like
        1D array of values of the signal corresponding to the given `times`.
        Will be resized to the size of `times` by zero-padding or truncating
        as necessary.
    value_type
        Type of signal, representing the units of the values. Values should be
        from the ``Signal.Type`` enum, but integer or string values may
        work if carefully chosen. ``Signal.Type.undefined`` by default.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    """
    class Type(Enum):
        """
        Enum containing possible types (units) for signal values.

        Attributes
        ----------
        voltage
        field
        power
        unknown, undefined

        """
        unknown = 0
        undefined = 0
        voltage = 1
        field = 2
        power = 3

    def __init__(self, times, values, value_type=None):
        self.times = np.array(times)
        len_diff = len(times)-len(values)
        if len_diff>0:
            self.values = np.concatenate((values, np.zeros(len_diff)))
        else:
            self.values = np.array(values[:len(times)])
        self.value_type = value_type

    def __add__(self, other):
        """
        Adds two signals by adding their values at each time.

        Adding ``Signal`` objects is only allowed when they have identical
        ``times`` arrays, and their ``value_type``s are compatible. This means
        that the ``value_type``s must be the same, or one must be ``undefined``
        which will be coerced to the other ``value_type``.

        Raises
        ------
        TypeError
            If the other object in the addition is not a ``Signal``.
        ValueError
            If the other ``Signal`` has different ``times`` or ``value_type``.

        """
        if not(isinstance(other, Signal)):
            raise TypeError("Can't add object with type"
                            +str(type(other))+" to a signal")
        if not(np.array_equal(self.times, other.times)):
            raise ValueError("Can't add signals with different times")
        if (self.value_type!=self.Type.undefined and
                other.value_type!=self.Type.undefined and
                self.value_type!=other.value_type):
            raise ValueError("Can't add signals with different value types")

        if self.value_type==self.Type.undefined:
            value_type = other.value_type
        else:
            value_type = self.value_type

        return Signal(self.times, self.values+other.values,
                      value_type=value_type)

    def __radd__(self, other):
        """
        Allows for adding Signal object to 0.

        Since the python ``sum`` function starts by adding the first element
        to 0, to use ``sum`` with ``Signal`` objects we need to be able to add
        a ``Signal`` object to 0. If adding to anything else, raise the usual
        error.

        """
        if other!=0:
            raise TypeError("unsupported operand type(s) for +: '"+
                            str(type(other))+"' and 'Signal'")

        return self

    @property
    def value_type(self):
        """
        Type of signal, representing the units of the values.

        Should always be a value from the ``Signal.Type`` enum. Setting with
        integer or string values may work if carefully chosen.

        """
        return self._value_type

    @value_type.setter
    def value_type(self, val_type):
        if val_type is None:
            self._value_type = self.Type.undefined
        else:
            self._value_type = get_from_enum(val_type, self.Type)

    @property
    def dt(self):
        """The time spacing of the `times` array, or ``None`` if invalid."""
        try:
            return self.times[1]-self.times[0]
        except IndexError:
            return None

    @property
    def envelope(self):
        """The envelope of the signal by Hilbert transform."""
        analytic_signal = scipy.signal.hilbert(self.values)
        return np.abs(analytic_signal)

    def resample(self, n):
        """
        Resamples the signal into n points in the same time range, in-place.

        Parameters
        ----------
        n : int
            The number of points into which the signal should be resampled.

        """
        if n==len(self.times):
            return

        self.times = np.linspace(self.times[0], self.times[-1], n)
        self.values = scipy.signal.resample(self.values, n)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times for which to define the new signal.

        Returns
        -------
        Signal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Interpolates the values of the ``Signal`` object across `new_times`,
        extrapolating with zero values on the left and right.

        """
        new_values = np.interp(new_times, self.times, self.values,
                               left=0, right=0)
        return Signal(new_times, new_values, value_type=self.value_type)


    @property
    def spectrum(self):
        """The FFT complex spectrum values of the signal."""
        return scipy.fftpack.fft(self.values)

    @property
    def frequencies(self):
        """The FFT frequencies of the signal."""
        return scipy.fftpack.fftfreq(n=len(self.values), d=self.dt)

    def filter_frequencies(self, freq_response, force_real=False):
        """
        Apply the given frequency response function to the signal, in-place.

        For the given response function, multiplies the response into the
        frequency domain of the signal. If the filtered signal is forced to be
        real, the positive-frequency response is mirrored into the negative
        frequencies by complex conjugation.

        Parameters
        ----------
        freq_response : function
            Response function taking a freqeuncy (or array of frequencies) and
            returning the corresponding complex gain(s).
        force_real : boolean
            If ``True``, complex conjugation is used on the positive-frequency
            response to force the filtered signal to be real-valued. Otherwise
            the frequency response is left alone and any imaginary parts of the
            filtered signal are thrown out.

        Warns
        -----
        Raises a warning if the maximum value of the imaginary part of the
        filtered signal was greater than 1e-5 times the maximum value of the
        real part, indicating that there was significant signal lost when
        discarding the imaginary part.

        """
        # Zero-pad the signal so the filter doesn't cause the resulting
        # signal to wrap around the end of the time array
        vals = np.concatenate((self.values, np.zeros(len(self.values))))
        spectrum = scipy.fftpack.fft(vals)
        freqs = scipy.fftpack.fftfreq(n=2*len(self.values), d=self.dt)
        if force_real:
            true_freqs = np.array(freqs)
            freqs = np.abs(freqs)

        # Attempt to evaluate all responses in one function call
        try:
            responses = np.array(freq_response(freqs), dtype=np.complex_)
        # Otherwise evaluate responses one at a time
        except (TypeError, ValueError):
            logger.debug("Frequency response function %r could not be "+
                         "evaluated for multiple frequencies at once",
                         freq_response)
            responses = np.zeros(len(spectrum), dtype=np.complex_)
            for i, f in enumerate(freqs):
                responses[i] = freq_response(f)

        # To make the filtered signal real, mirror the positive frequency
        # response into the negative frequencies, making the real part even
        # (done above) and the imaginary part odd (below)
        if force_real:
            responses.imag[true_freqs<0] *= -1

        filtered_vals = scipy.fftpack.ifft(responses*spectrum)
        self.values = np.real(filtered_vals[:len(self.times)])

        # Issue a warning if there was significant signal in the (discarded)
        # imaginary part of the filtered values
        if np.any(np.imag(filtered_vals[:len(self.times)]) >
                  np.max(self.values) * 1e-5):
            msg = ("Significant signal amplitude was lost when forcing the "+
                   "signal values to be real after applying the frequency "+
                   "filter '%s'. This may be avoided by making sure the "+
                   "filter being used is properly defined for negative "+
                   "frequencies, or by passing force_real=True to the "+
                   "Signal.filter_frequencies function.")
            logger.warning(msg, freq_response.__name__)



class EmptySignal(Signal):
    """
    Class for signal with zero amplitude (all values = 0).

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.

    """
    def __init__(self, times, value_type=None):
        super().__init__(times, np.zeros(len(times)), value_type=value_type)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times for which to define the new signal.

        Returns
        -------
        EmtpySignal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Since the ``EmptySignal`` always has zero values, the returned signal
        will also have all zero values.

        """
        return EmptySignal(new_times, value_type=self.value_type)


class FunctionSignal(Signal):
    """
    Class for signals generated by a function.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    function : function
        Function which evaluates the corresponding value(s) for a given time or
        array of times.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    function : function
        Function to evaluate the signal values at given time(s).
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.
    EmptySignal : Class for signal with zero amplitude.

    """
    def __init__(self, times, function, value_type=None):
        self.times = np.array(times)
        self.function = function
        # Attempt to evaluate all values in one function call
        try:
            values = self.function(self.times)
        # Otherwise evaluate values one at a time
        except (ValueError, TypeError):
            values = []
            for t in self.times:
                values.append(self.function(t))

        super().__init__(times, values, value_type=value_type)

    def with_times(self, new_times):
        """
        Returns a representation of this signal over a different times array.

        Parameters
        ----------
        new_times : array_like
            1D array of times for which to define the new signal.

        Returns
        -------
        FunctionSignal
            A representation of the original signal over the `new_times` array.

        Notes
        -----
        Leverages knowledge of the function that creates the signal to properly
        recalculate exact (not interpolated) values for the new times.

        """
        return FunctionSignal(new_times, self.function,
                              value_type=self.value_type)



class SlowAskaryanSignal(Signal):
    """
    Class for generating Askaryan signals according to ARVZ parameterization.

    Stores the time-domain information for an Askaryan electric field (V/m).
    The amplitude of the pulse is calculated at 1 meter from the shower, so a
    1/R effect (where R is the distance from source to observer) must be
    applied to produce the proper signal amplitude at the observation point.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    energy : float
        Energy (GeV) of the particle shower producing the pulse.
    theta : float
        Observation angle (radians) measured relative to the shower axis.
    n : float, optional
        Index of refraction at the location of the shower.
    t0 : float, optional
        Pulse offset time (s), i.e. time at which the shower takes place.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type.field
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    energy : float
        Energy (GeV) of the particle shower producing the pulse.
    vector_potential : ndarray
        1D array of vector potential (V*s/m) of the pulse
    dt
    frequencies
    spectrum
    envelope

    Warns
    -----
    Raises warning that this class is essentially deprecated and
    ARVZAskaryanSignal should be used instead.

    Warnings
    --------
    This class is essentially deprecated. ``ARVZAskaryanSignal`` is faster and
    more accurate. This class is kept for reference only and shouldn't be used.

    See Also
    --------
    Signal : Base class for time-domain signals.
    ARVZAskaryanSignal : Faster class for ARVZ Askaryan parameterization.

    Notes
    -----
    Calculates the Askaryan signal based on the ARVZ (Alvarez-Muniz,
    Romero-Wolf, Zas) parameterization [1]_. Uses a simplified Heitler model for
    the shower profile [2]_. Calculates the vector potential directly, and then
    numerically differentiates to get the electric field. This method is
    considerably slower (~1000x) than the convolution method in
    ``ARVZAskaryanSignal``, and seems to be backwards in time as well. The
    class is kept for reference only and shouldn't be used.

    References
    ----------
    .. [1] Jaime Alvarez-Muñiz et al, "Practical and accurate calculations
        of Askaryan radiation." Physical Review D **84**, 103003 (2011).
    .. [2] K.D. de Vries et al, "On the feasibility of RADAR detection of
        high-energy neutrino-induced showers in ice." Astropart. Phys.
        **60**, 25-31 (2015).

    """
    def __init__(self, times, energy, theta, n=1.78, t0=0):
        logger.warning("SlowAskaryanSignal is deprecated, use "+
                       "ARVZAskaryanSignal (aliased to AskaryanSignal) instead")
        # Calculation of pulse based on https://arxiv.org/pdf/1106.6283v3.pdf
        # Vector potential is given by:
        #   A(theta,t) = integral(Q(z) * RAC(t-z(1-n*cos(theta))/c))
        #                * sin(theta) / sin(theta_c) / R / integral(Q(z))
        self.energy = energy
        self.vector_potential = np.zeros(len(times))

        # Conversion factor for z in RAC
        z_to_t = (1 - n*np.cos(theta))/3e8

        # Integrals of z go from z_min to z_max with n_z steps
        z_min = 0
        z_max = 2.5*self.max_length()
        n_z = 1000
        z_vals, dz = np.linspace(z_min, z_max, n_z, endpoint=False,
                                 retstep=True)

        # Q(z) is the same for every time
        Q = np.zeros(n_z)
        for i, z in enumerate(z_vals):
            Q[i] = self.charge_profile(z)

        # Calculate RAC and integral(Q*RAC) at each time
        for i, t in enumerate(times):
            RA_C = np.zeros(n_z)
            for j, z in enumerate(z_vals):
                RA_C[j] = self.RAC(t - t0 - z*z_to_t)
            self.vector_potential[i] = np.trapz(Q*RA_C, dx=dz)

        # Calculate LQ_tot (the excess longitudinal charge along the shower)
        LQ_tot = np.trapz(Q, dx=dz)

        # Calculate sin(theta_c) = sqrt(1-cos^2(theta_c)) = sqrt(1-1/n^2)
        sin_theta_c = np.sqrt(1 - 1/n**2)

        # Scale the integral by the necessary factors
        self.vector_potential *= np.sin(theta) / sin_theta_c / LQ_tot

        # Calculate electric field by taking derivative of vector potential
        dt = times[1] - times[0]
        values = -np.diff(self.vector_potential) / dt

        super().__init__(times, values, value_type=self.Type.field)


    def RAC(self, time):
        """
        Calculates R*A_C at the given time.

        The R*A_C value is the observation distance R (m) times the vector
        potential (V*s/m) at the Cherenkov angle.

        Parameters
        ----------
        time : float
            Time (s) at which to calculate the R*A_C value.

        Returns
        -------
        float
            The R*A_C value (V*s) at the given time.

        Notes
        -----
        Based on equation 16 of the ARVZ paper [1]_.

        References
        ----------
        .. [1] Jaime Alvarez-Muñiz et al, "Practical and accurate calculations
            of Askaryan radiation." Physical Review D **84**, 103003 (2011).

        """
        # Get absolute value of time in nanoseconds
        ta = np.abs(time) * 1e9
        if time>=0:
            return (-4.5e-17 * self.energy
                    * (np.exp(-ta/0.057) + (1+2.87*ta)**-3))
        else:
            return (-4.5e-17 * self.energy
                    * (np.exp(-ta/0.030) + (1+3.05*ta)**-3.5))

    def charge_profile(self, z, density=0.92, crit_energy=7.86e-2,
                       rad_length=36.08):
        """
        Calculates the EM shower longitudinal charge profile.

        The longitudinal charge profile is calculated for a given distance,
        density, critical energy, and electron radiation length in ice.

        Parameters
        ----------
        z : float
            Distance (m) along the shower at which to calculate the charge.
        density : float, optional
            Density (g/cm^3) of ice.
        crit_energy : float, optional
            Critical energy (GeV) for shower formation.
        rad_length : float, optional
            Electron radiation length (g/cm^2) in ice.

        Returns
        -------
        float
            The charge (C) at the given distance along the shower.

        Notes
        -----
        Profile calculated by a simplified Heitler model ased on equations 24
        and 25 of the radar feasibility paper [1]_.

        References
        ----------
        .. [1] K.D. de Vries et al, "On the feasibility of RADAR detection of
            high-energy neutrino-induced showers in ice." Astropart. Phys.
            **60**, 25-31 (2015).

        """
        # Based on Heitler model as described in appendix of
        # https://arxiv.org/pdf/1312.4331v2.pdf
        if z<=0 or self.energy<=crit_energy:
            return 0

        # Depth calculated by "integrating" the density along the shower path
        # (in g/cm^2)
        x = 100 * z * density
        x_ratio = x / rad_length
        e_ratio = self.energy / crit_energy

        # Shower age
        s = 3 * x_ratio / (x_ratio + 2*np.log(e_ratio))

        # Number of particles
        N = (0.31 * np.exp(x_ratio * (1 - 1.5*np.log(s)))
             / np.sqrt(np.log(e_ratio)))

        return N * 1.602e-19

    def max_length(self, density=0.92, crit_energy=7.86e-2, rad_length=36.08):
        """
        Calculates the maximum length (or shower depth) of an EM shower.

        The maximum length of an EM shower is calculated for a given density,
        critical energy, and electron radiation length in ice.

        Parameters
        ----------
        density : float, optional
            Density (g/cm^3) of ice.
        crit_energy : float, optional
            Critical energy (GeV) for shower formation.
        rad_length : float, optional
            Electron radiation length (g/cm^2) in ice.

        Returns
        -------
        float
            The maximum length (m) of an EM shower.

        """
        # Maximum depth in g/cm^2
        x_max = rad_length * np.log(self.energy / crit_energy) / np.log(2)

        return 0.01 * x_max / density


class ARVZAskaryanSignal(Signal):
    """
    Class for generating Askaryan signals according to ARVZ parameterization.

    Stores the time-domain information for an Askaryan electric field (V/m)
    produced by the electromagnetic shower initiated by a neutrino.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    particle : Particle
        ``Particle`` object responsible for the shower which produces the
        Askaryan signal. Should have an ``energy`` in GeV, ``vertex`` in m,
        and ``id``, plus an ``interaction`` with a ``kind``.
    viewing_angle : float
        Observation angle (radians) measured relative to the shower axis.
    viewing_distance : float, optional
        Distance (m) between the shower vertex and the observation point (along
        the ray path).
    ice_model : optional
        The ice model to be used for describing the index of refraction of the
        medium.
    t0 : float, optional
        Pulse offset time (s), i.e. time at which the shower takes place.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type.field
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    energy : float
        Energy (GeV) of the particle shower producing the pulse.
    em_frac : float
        Fraction of the particle energy which goes into the electromagnetic
        cascade.
    had_frac : float
        Fraction of the particle energy which goes into the hadronic cascade.
    vector_potential
    dt
    frequencies
    spectrum
    envelope

    Raises
    ------
    ValueError
        If the `particle` object is not a neutrino or antineutrino with a
        charged-current or neutral-current interaction.

    See Also
    --------
    Signal : Base class for time-domain signals.

    Notes
    -----
    Calculates the Askaryan signal based on the ARVZ parameterization [1]_.
    Uses a Heitler model for the electromagnetic shower profile [2]_.
    Calculates the electric field from the vector potential using the
    convolution method outlined in section 4 of the ARVZ paper, which results
    in the most efficient calculation of the parameterization.

    References
    ----------
    .. [1] Jaime Alvarez-Muñiz et al, "Practical and accurate calculations
        of Askaryan radiation." Physical Review D **84**, 103003 (2011).
    .. [2] K.D. de Vries et al, "On the feasibility of RADAR detection of
        high-energy neutrino-induced showers in ice." Astropart. Phys.
        **60**, 25-31 (2015).

    """
    def __init__(self, times, particle, viewing_angle, viewing_distance=1,
                 ice_model=IceModel, t0=0):
        # Calculation of pulse based on https://arxiv.org/pdf/1106.6283v3.pdf
        # Vector potential is given by:
        #   A(theta,t) = convolution(Q(z(1-n*cos(theta))/c)),
        #                            RAC(z(1-n*cos(theta))/c))
        #                * sin(theta) / sin(theta_c) / R / integral(Q(z))
        #                * c / (1-n*cos(theta))

        # Theta should represent the angle from the shower axis, and so should
        # always be positive
        theta = np.abs(viewing_angle)

        if theta > np.pi:
            raise ValueError("Angles greater than 180 degrees not supported")

        # Calculate shower energy based on particle energy and inelasticity
        if (particle.interaction.kind ==
                particle.interaction.Type.neutral_current):
            self.em_frac = 0
            self.had_frac = particle.inelasticity
        elif (particle.interaction.kind ==
              particle.interaction.Type.charged_current):
            if (particle.id==particle.Type.electron_neutrino or
                    particle.id==particle.Type.electron_antineutrino):
                self.em_frac = 1 - particle.inelasticity
                self.had_frac = particle.inelasticity
            elif (particle.id==particle.Type.muon_neutrino or
                  particle.id==particle.Type.muon_antineutrino):
                self.em_frac = 0
                self.had_frac = particle.inelasticity
            elif (particle.id==particle.Type.tau_neutrino or
                  particle.id==particle.Type.tau_antineutrino):
                self.em_frac = 0
                self.had_frac = particle.inelasticity
            else:
                raise ValueError("Particle type not supported")
        else:
            raise ValueError("Interaction type not supported")

        self.energy = particle.energy * self.em_frac

        # Calculate index of refraction at the shower position for the
        # Cherenkov angle calculation and others
        n = ice_model.index(particle.vertex[2])

        # Conversion factor from z to t for RAC:
        # (1-n*cos(theta)) / c
        z_to_t = (1 - n*np.cos(theta))/3e8

        # Calculate the time step and the corresponding z-step
        dt = times[1] - times[0]

        # Calculate the corresponding z-step (dz = dt / z_to_t)
        # If the z-step is too large compared to the expected maximum shower
        # length, then the result will be bad. Set dt_divider so that
        # dz / max_length <= 0.1 (with dz=dt/z_to_t)
        dt_divider = int(np.abs(10*dt/self.max_length()/z_to_t)) + 1
        dz = dt / dt_divider / z_to_t
        if dt_divider!=1:
            logger.debug("z-step of %g too large; dt_divider changed to %g",
                         dt / z_to_t, dt_divider)

        # Create the charge-profile array up to 2.5 times the nominal
        # maximum shower length (to reduce errors)
        z_max = 2.5*self.max_length()
        n_Q = int(np.abs(z_max/dz))
        z_Q_vals = np.arange(n_Q) * np.abs(dz)
        Q = np.zeros(n_Q)
        for i, z in enumerate(z_Q_vals):
            Q[i] = self.charge_profile(z)

        # Fail gracefully if the energy is less than the critical energy for
        # shower formation (i.e. all Q values are zero)
        if np.all(Q==0) and len(Q)>0:
            super().__init__(times, np.zeros(len(times)))
            return

        # Calculate RAC at a specific number of t values (n_RAC) determined so
        # that the full convolution will have the same size as the times array,
        # when appropriately rescaled by dt_divider.
        # If t_RAC_vals does not include a reasonable range around zero
        # (typically because n_RAC is too small), errors occur. In that case
        # extra points are added at the beginning and/or end of RAC.
        # If n_RAC is too large, the convolution can take a very long time.
        # In that case, points are removed from the beginning and/or end of RAC.
        t_tolerance = 10e-9
        t_start = times[0] - t0
        n_extra_beginning = int((t_start+t_tolerance)/dz/z_to_t) + 1
        n_extra_end = (int((t_tolerance-t_start)/dz/z_to_t) + 1
                       + n_Q - len(times)*dt_divider)
        n_RAC = (len(times)*dt_divider + 1 - n_Q
                 + n_extra_beginning + n_extra_end)
        t_RAC_vals = (np.arange(n_RAC) * dz * z_to_t
                      + t_start - n_extra_beginning * dz * z_to_t)
        RA_C = np.zeros(n_RAC)
        for i, t in enumerate(t_RAC_vals):
            RA_C[i] = self.RAC(t)

        # Convolve Q and RAC to get unnormalized vector potential
        if n_Q*n_RAC>1e6:
            logger.debug("convolving %i Q points with %i RA_C points",
                         n_Q, n_RAC)
        convolution = scipy.signal.convolve(Q, RA_C, mode='full')

        # Adjust convolution by zero-padding or removing values according to
        # the values added/removed at the beginning and end of RA_C
        if n_extra_beginning<0:
            convolution = np.concatenate((np.zeros(-n_extra_beginning),
                                          convolution))
        else:
            convolution = convolution[n_extra_beginning:]
        if n_extra_end<=0:
            convolution = np.concatenate((convolution,
                                          np.zeros(-n_extra_end)))
        else:
            convolution = convolution[:-n_extra_end]

        # Reduce the number of values in the convolution based on the dt_divider
        # so that the number of values matches the length of the times array.
        # It's possible that this should be using scipy.signal.resample instead
        # TODO: Figure that out
        convolution = convolution[::dt_divider]

        # Calculate LQ_tot (the excess longitudinal charge along the shower)
        LQ_tot = np.trapz(Q, dx=dz)

        # Calculate sin(theta_c) = sqrt(1-cos^2(theta_c)) = sqrt(1-1/n^2)
        sin_theta_c = np.sqrt(1 - 1/n**2)

        # Scale the convolution by the necessary factors to get the true
        # vector potential A
        # z_to_t and dt_divider are divided based on trial and error to correct
        # the normalization. They are not proven nicely like the other factors
        A = (convolution * -1 * np.sin(theta) / sin_theta_c / LQ_tot
             / z_to_t / dt_divider)

        # Not sure why, but multiplying A by -dt is necessary to fix
        # normalization and dependence of amplitude on time spacing.
        # Since E = -dA/dt = np.diff(A) / -dt, we can skip multiplying
        # and later dividing by dt to save a little computational effort
        # (at the risk of more cognitive effort when deciphering the code)
        # So, to clarify, the above statement should have "* -dt" at the end
        # to be the true value of A, and the below would then have "/ -dt"

        # Calculate electric field by taking derivative of vector potential,
        # and divide by the viewing distance (R)
        values = np.diff(A) / viewing_distance

        # Note that although len(values) = len(times)-1 (because of np.diff),
        # the Signal class is desinged to handle this by zero-padding the values
        super().__init__(times, values, value_type=self.Type.field)


    @property
    def vector_potential(self):
        """
        The vector potential of the signal.

        Recovered from the electric field, mostly just for testing purposes.

        """
        return np.cumsum(np.concatenate(([0],self.values)))[:-1] * -self.dt


    def RAC(self, time):
        """
        Calculates R*A_C at the given time.

        The R*A_C value is the observation distance R (m) times the vector
        potential (V*s/m) at the Cherenkov angle.

        Parameters
        ----------
        time : float
            Time (s) at which to calculate the R*A_C value.

        Returns
        -------
        float
            The R*A_C value (V*s) at the given time.

        Notes
        -----
        Based on equation 16 of the ARVZ paper [1]_.

        References
        ----------
        .. [1] Jaime Alvarez-Muñiz et al, "Practical and accurate calculations
            of Askaryan radiation." Physical Review D **84**, 103003 (2011).

        """
        # Get absolute value of time in nanoseconds
        ta = np.abs(time) * 1e9
        if time>=0:
            return (-4.5e-17 * self.energy
                    * (np.exp(-ta/0.057) + (1+2.87*ta)**-3))
        else:
            return (-4.5e-17 * self.energy
                    * (np.exp(-ta/0.030) + (1+3.05*ta)**-3.5))

    def charge_profile(self, z, density=0.92, crit_energy=7.86e-2,
                       rad_length=36.08):
        """
        Calculates the EM shower longitudinal charge profile.

        The longitudinal charge profile is calculated for a given distance,
        density, critical energy, and electron radiation length in ice.

        Parameters
        ----------
        z : float
            Distance (m) along the shower at which to calculate the charge.
        density : float, optional
            Density (g/cm^3) of ice.
        crit_energy : float, optional
            Critical energy (GeV) for shower formation.
        rad_length : float, optional
            Electron radiation length (g/cm^2) in ice.

        Returns
        -------
        float
            The charge (C) at the given distance along the shower.

        Notes
        -----
        Profile calculated by a simplified Heitler model ased on equations 24
        and 25 of the radar feasibility paper [1]_.

        References
        ----------
        .. [1] K.D. de Vries et al, "On the feasibility of RADAR detection of
            high-energy neutrino-induced showers in ice." Astropart. Phys.
            **60**, 25-31 (2015).

        """
        if z<=0 or self.energy<=crit_energy:
            return 0

        # Depth calculated by "integrating" the density along the shower path
        # (in g/cm^2)
        x = 100 * z * density
        x_ratio = x / rad_length
        e_ratio = self.energy / crit_energy

        # Shower age
        s = 3 * x_ratio / (x_ratio + 2*np.log(e_ratio))

        # Number of particles
        N = (0.31 * np.exp(x_ratio * (1 - 1.5*np.log(s)))
             / np.sqrt(np.log(e_ratio)))

        return N * 1.602e-19

    def max_length(self, density=0.92, crit_energy=7.86e-2, rad_length=36.08):
        """
        Calculates the maximum length (or shower depth) of an EM shower.

        The maximum length of an EM shower is calculated for a given density,
        critical energy, and electron radiation length in ice.

        Parameters
        ----------
        density : float, optional
            Density (g/cm^3) of ice.
        crit_energy : float, optional
            Critical energy (GeV) for shower formation.
        rad_length : float, optional
            Electron radiation length (g/cm^2) in ice.

        Returns
        -------
        float
            The maximum length (m) of an EM shower.

        """
        # Maximum depth in g/cm^2
        x_max = rad_length * np.log(self.energy / crit_energy) / np.log(2)

        return 0.01 * x_max / density


# ARVZAskaryanSignal result now matches SlowAskaryanSignal.
# In fact, ARVZAskaryanSignal performs much better:
#   Runs about 1000x faster
#   Causal, whereas SlowAskaryanSignal pulse seems to be backwards in time
#   Smooth pulse; no artifacts from integration errors
AskaryanSignal = ARVZAskaryanSignal



class GaussianNoise(Signal):
    """
    Class for gaussian noise signals with standard deviation sigma.

    Calculates each time value independently from a normal distribution.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    values : array_like
        1D array of values of the signal corresponding to the given `times`.
        Will be resized to the size of `times` by zero-padding or truncating.
    value_type
        Type of signal, representing the units of the values. Must be from the
        ``Signal.Type`` Enum.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type.voltage
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    dt
    frequencies
    spectrum
    envelope

    See Also
    --------
    Signal : Base class for time-domain signals.

    """
    def __init__(self, times, sigma):
        self.sigma = sigma
        values = np.random.normal(0, self.sigma, size=len(times))
        super().__init__(times, values, value_type=self.Type.voltage)


class ThermalNoise(FunctionSignal):
    """
    Class for thermal Rayleigh noise signals.

    The Rayleigh thermal noise is calculated in a given frequency band with
    flat or otherwise specified amplitude and random phase at some number of
    frequencies. Values are scaled to a provided or calculated RMS voltage.

    Parameters
    ----------
    times : array_like
        1D array of times for which the signal is defined.
    f_band : array_like
        Array of two elements denoting the frequency band (Hz) of the noise.
        The first element should be smaller than the second.
    f_amplitude : float or function, optional
        The frequency-domain amplitude of the noise. If ``float``, then all
        frequencies will have the same amplitude. If ``function``, then the
        function is evaluated at each frequency to determine its amplitude.
    rms_voltage : float, optional
        The RMS voltage (V) of the noise. If specified, this value will be used
        instead of the RMS voltage calculated from the values of `temperature`
        and `resistance`.
    temperature : float, optional
        The thermal noise temperature (K). Used in combination with the value
        of `resistance` to calculate the RMS voltage of the noise.
    resistance : float, optional
        The resistance (ohm) for the noise. Used in combination with the value
        of `temperature` to calculate the RMS voltage of the noise.
    n_freqs : int, optional
        The number of frequencies within the frequency band to use to calculate
        the noise signal. By default determines the number of frequencies based
        on the FFT bin size of `times`.

    Attributes
    ----------
    times, values : ndarray
        1D arrays of times and corresponding values which define the signal.
    value_type : Signal.Type.voltage
        Type of signal, representing the units of the values.
    Type : Enum
        Different value types available for `value_type` of signal objects.
    function : function
        Function to evaluate the signal values at given time(s).
    f_min : float
        Minimum frequency of the noise frequency band.
    f_max : float
        Maximum frequency of the noise frequency band.
    freqs, amps, phases : ndarray
        The frequencies used to define the noise signal and their corresponding
        amplitudes and phases.
    rms : float
        The RMS value of the noise signal.
    dt
    frequencies
    spectrum
    envelope

    Warnings
    --------
    Since this class inherits from ``FunctionSignal``, its ``with_times``
    method will properly extrapolate noise outside of the provided times. Be
    warned however that outside of the original signal times the noise signal
    will be highly periodic. Since the default number of frequencies used is
    based on the FFT bin size of `times`, the period of the noise signal is
    actually the length of `times`. As a result if you are planning on
    extrapolating the noise signal, increasing the number of frequencies used
    is strongly recommended.

    Raises
    ------
    ValueError
        If the RMS voltage cannot be calculated (i.e. `rms_voltage` or both
        `temperature` and `resistance` are ``None``).

    See Also
    --------
    FunctionSignal : Class for signals generated by a function.

    Notes
    -----
    Calculation of the noise signal is based on the Rayleigh noise model used
    by ANITA [1]_.

    References
    ----------
    .. [1] A. Connolly et al, ANITA Note #76, "Thermal Noise Studies: Toward A
        Time-Domain Model of the ANITA Trigger."
        https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

    """
    def __init__(self, times, f_band, f_amplitude=1, rms_voltage=None,
                 temperature=None, resistance=None, n_freqs=0):
        # Calculation based on Rician (Rayleigh) noise model for ANITA:
        # https://www.phys.hawaii.edu/elog/anita_notes/060228_110754/noise_simulation.ps

        self.f_min, self.f_max = f_band
        if self.f_min>=self.f_max:
            raise ValueError("Frequency band must have smaller frequency as "+
                             "first value and larger frequency as second value")
        # If number of frequencies is unspecified (or invalid),
        # determine based on the FFT bin size of the times array
        if n_freqs<1:
            n_freqs = (self.f_max - self.f_min) * (times[-1] - times[0])
            # Broken out into steps to ease understanding:
            #   duration = times[-1] - times[0]
            #   f_bin_size = 1 / duration
            #   n_freqs = (self.f_max - self.f_min) / f_bin_size

        # If number of frequencies is still zero (e.g. len(times)==1),
        # force it to 1
        if n_freqs<1:
            n_freqs = 1

        self.freqs = np.linspace(self.f_min, self.f_max, int(n_freqs),
                                 endpoint=False)

        # Allow f_amplitude to be either a function or a single value
        if callable(f_amplitude):
            self.amps = [f_amplitude(f) for f in self.freqs]
        else:
            self.amps = np.full(len(self.freqs), f_amplitude, dtype="float64")
            # If the frequency range extends to zero, force the zero-frequency
            # (DC) amplitude to zero
            if 0 in self.freqs:
                self.amps[np.where(self.freqs==0)[0]] = 0

        self.phases = np.random.rand(len(self.freqs)) * 2*np.pi

        if rms_voltage is not None:
            self.rms = rms_voltage
        elif temperature is not None and resistance is not None:
            # RMS voltage = sqrt(4 * kB * T * R * bandwidth)
            self.rms = np.sqrt(4 * 1.38e-23 * temperature * resistance
                               * (self.f_max - self.f_min))
        else:
            raise ValueError("Either RMS voltage or temperature and resistance"+
                             " must be provided to calculate noise amplitude")

        def f(ts):
            """Set the time-domain signal by adding sinusoidal signals of each
            frequency with the corresponding phase."""
            # This method is nicer than an inverse fourier transform because
            # results are consistant for differing ranges of ts around the same
            # time. The price payed is that the fft would be an order of
            # magnitude faster, but not reproducible for a slightly different
            # time array.
            values = sum(amp * np.cos(2*np.pi*freq * ts + phase)
                         for freq, amp, phase
                         in zip(self.freqs, self.amps, self.phases))

            # Normalization calculated by guess-and-check,
            # but seems to work fine
            # normalization = np.sqrt(2/len(self.freqs))
            values *= np.sqrt(2/len(self.freqs))

            # So far, the units of the values are V/V_rms, so multiply by the
            # rms voltage:
            values *= self.rms

            return values

        super().__init__(times, function=f, value_type=self.Type.voltage)
