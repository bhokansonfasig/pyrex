"""
Module containing customized antenna classes for IREX.

The IREX antennas are based around existing ARA antennas with an extra
envelope circuit applied in the front-end, designed to reduce power
consumption and the amount of digitized information.

"""

import numpy as np
import scipy.constants
import scipy.signal
from pyrex.signals import Signal
from pyrex.antenna import Antenna

from pyrex.custom.ara.antenna import (ARAAntennaSystem, HPOL_RESPONSE_DATA,
                                      VPOL_RESPONSE_DATA)
from .frontends import (pyspice, spice_circuits,
                        basic_envelope_model, bridge_rectifier_envelope_model)


class DipoleTester(Antenna):
    """
    Dipole antenna for IREX testing.

    Stores the attributes of an antenna as well as handling receiving,
    processing, and storing signals and adding noise. Uses a first-order
    butterworth filter for the frequency response.

    Parameters
    ----------
    position : array_like
        Vector position of the antenna.
    center_frequency : float
        Tuned frequency (Hz) of the dipole.
    bandwidth : float
        Bandwidth (Hz) of the antenna.
    temperature : float
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    effective_height : float, optional
        Effective length (m) of the antenna. By default calculated by the tuned
        `center_frequency` of the dipole.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    z_axis : ndarray
        Vector direction of the z-axis of the antenna.
    x_axis : ndarray
        Vector direction of the x-axis of the antenna.
    antenna_factor : float
        Antenna factor used for converting fields to voltages.
    efficiency : float
        Antenna efficiency applied to incoming signal values.
    threshold : float, optional
        Voltage threshold (V) above which signals will trigger.
    effective_height : float, optional
        Effective length of the antenna. By default calculated by the tuned
        `center_frequency` of the dipole.
    filter_coeffs : tuple of ndarray
        Coefficients of transfer function for butterworth bandpass filter to be
        used for frequency response.
    noisy : boolean
        Whether or not the antenna should add noise to incoming signals.
    unique_noises : int
        The number of expected noise waveforms needed for each received signal
        to have its own noise.
    freq_range : array_like
        The frequency band in which the antenna operates (used for noise
        production).
    temperature : float or None
        The noise temperature (K) of the antenna. Used in combination with
        `resistance` to calculate the RMS voltage of the antenna noise.
    resistance : float or None
        The noise resistance (ohm) of the antenna. Used in combination with
        `temperature` to calculate the RMS voltage of the antenna noise.
    noise_rms : float or None
        The RMS voltage (V) of the antenna noise. If not ``None``, this value
        will be used instead of the RMS voltage calculated from the values of
        `temperature` and `resistance`.
    signals : list of Signal
        The signals which have been received by the antenna.
    is_hit
    is_hit_mc_truth
    waveforms
    all_waveforms

    """
    def __init__(self, position, center_frequency, bandwidth, temperature,
                 resistance, orientation=(0,0,1), effective_height=None,
                 noisy=True, unique_noise_waveforms=10):
        if effective_height is None:
            # Calculate length of half-wave dipole
            self.effective_height = scipy.constants.c / center_frequency / 2
        else:
            self.effective_height = effective_height

        # Get the critical frequencies in Hz
        f_low = center_frequency - bandwidth/2
        f_high = center_frequency + bandwidth/2

        # Get arbitrary x-axis orthogonal to orientation
        tmp_vector = np.zeros(3)
        while np.array_equal(np.cross(orientation, tmp_vector), (0,0,0)):
            tmp_vector = np.random.rand(3)
        ortho = np.cross(orientation, tmp_vector)
        # Note: ortho is not normalized, but will be normalized by Antenna init

        super().__init__(position=position, z_axis=orientation, x_axis=ortho,
                         antenna_factor=1/self.effective_height,
                         freq_range=(f_low, f_high),
                         temperature=temperature, resistance=resistance,
                         unique_noise_waveforms=unique_noise_waveforms,
                         noisy=noisy)

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)

    def frequency_response(self, frequencies):
        """
        Calculate the (complex) frequency response of the antenna.

        Dipole antenna frequency response is a first order butterworth bandpass
        filter in the antenna's frequency range.

        Parameters
        ----------
        frequencies : array_like
            1D array of frequencies at which to calculate gains.

        Returns
        -------
        array_like
            Complex gains in voltage for the given `frequencies`.

        """
        angular_freqs = np.array(frequencies) * 2*np.pi
        w, h = scipy.signal.freqs(self.filter_coeffs[0], self.filter_coeffs[1],
                                  angular_freqs)
        return h

    def directional_gain(self, theta, phi):
        """
        Calculate the (complex) directional gain of the antenna.

        Power gain of dipole antenna goes as sin(theta)^2, so electric field
        gain goes as sin(theta).

        Parameters
        ----------
        theta : float
            Polar angle (radians) from which a signal is arriving.
        phi : float
            Azimuthal angle (radians) from which a signal is arriving.

        Returns
        -------
        complex
            Complex gain in voltage for the given incoming angles.

        """
        return np.sin(theta)

    def polarization_gain(self, polarization):
        """
        Calculate the (complex) polarization gain of the antenna.

        Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis.

        Parameters
        ----------
        polarization : array_like
            Vector polarization direction of the signal.

        Returns
        -------
        complex
            Complex gain in voltage for the given signal polarization.

        """
        return np.vdot(self.z_axis, polarization)



class EnvelopeSystem(ARAAntennaSystem):
    """
    Antenna system extending ARA antennas with an envelope circuit.

    Consists of an ARA antenna with typical responses, front-end electronics,
    and amplifier clipping, but with an additional amplification and envelope
    circuit applied after all other front-end processing.

    Parameters
    ----------
    response_data : tuple of array_like
        Tuple containing the response data for the antenna along the theta
        and phi polarization directions. The first and second elements should
        contain 3-D arrays of the antenna response model in the theta and phi
        polarizations, respectively, as a function of frequency (axis 0),
        zenith (axis 1), and azimuth (axis 2). The remaining elements should be
        the values of the frequency, zenith, and azimuth axes, respectively.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float, optional
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    envelope_amplification : float, optional
        Amplification to be applied to the signal after the typical ARA front
        end, before the envelope circuit.
    envelope_method : {('hilbert', 'analytic', 'spice') + ('basic', 'biased',\
                        'doubler', 'bridge', 'log amp')}, optional
        String describing the circuit (and calculation method) to be used for
        envelope calculation. If the string contains "hilbert", the hilbert
        envelope is used. If the string contains "analytic", an analytic form
        is used to calculate the circuit output. If the string contains
        "spice", ``ngspice`` is used to calculate the circuit output. The
        default value "analytic" uses an analytic diode bridge circuit.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    envelope_amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    envelope_method : str
        String describing the circuit (and calculation method) to be used for
        envelope calculation.
    lead_in_time : float
        Lead-in time (s) required for the front end to equilibrate.
        Automatically added in before calculation of signals and waveforms.
    is_hit
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    See Also
    --------
    pyrex.custom.ara.antenna.ARAAntennaSystem : Antenna system extending base
                                                ARA antenna with front-end
                                                processing.
    pyrex.custom.ara.antenna.ARAAntenna : Antenna class to be used for ARA
                                          antennas.

    """
    lead_in_time = 25e-9

    def __init__(self, response_data, name, position, trigger_threshold,
                 time_over_threshold=0, orientation=(0,0,1), amplification=1,
                 amplifier_clipping=1, envelope_amplification=1,
                 envelope_method="analytic", noisy=True,
                 unique_noise_waveforms=10):
        super().__init__(response_data=response_data, name=name,
                         position=position, power_threshold=0,
                         orientation=orientation,
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)

        self.envelope_amplification = envelope_amplification

        self.trigger_threshold = trigger_threshold
        self.time_over_threshold = time_over_threshold

        self.envelope_method = envelope_method

    @property
    def _metadata(self):
        """Metadata dictionary for writing `EnvelopeSystem` information."""
        meta = super()._metadata
        meta.update({
            "lead_in_time": self.lead_in_time,
            "envelope_amplification": self.envelope_amplification,
            "envelope_method": self.envelope_method,
            "trigger_threshold": self.trigger_threshold,
            "time_over_threshold": self.time_over_threshold
        })
        return meta

    def make_envelope(self, signal):
        """
        Return the signal envelope based on the antenna's ``envelope_method``.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to apply the envelope.

        Returns
        -------
        Signal
            Signal processed by the envelope circuit.

        Raises
        ------
        ValueError
            If the antenna's ``envelope_method`` is invalid.
        ModuleNotFoundError
            If "spice" is in ``envelope_method`` and ``PySpice`` hasn't been
            installed.

        See Also
        --------
        pyrex.custom.irex.frontends.basic_envelope_model :
            Model of a basic diode-capacitor-resistor envelope circuit.
        pyrex.custom.irex.frontends.bridge_rectifier_envelope_model :
            Model of a diode bridge rectifier envelope circuit.

        """
        if "hilbert" in self.envelope_method:
            return Signal(signal.times, signal.envelope,
                          value_type=signal.value_type)

        elif "analytic" in self.envelope_method:
            if ("bridge" in self.envelope_method or
                    self.envelope_method=="analytic"):
                return bridge_rectifier_envelope_model(signal)
            elif "basic" in self.envelope_method:
                return basic_envelope_model(signal)
            else:
                raise ValueError("Only basic and bridge rectifier envelope "+
                                 "circuits are modeled analytically")

        elif "spice" in self.envelope_method:
            if not(pyspice.__available__):
                raise ModuleNotFoundError(pyspice.__modulenotfound__)

            if self.envelope_method=="spice":
                raise ValueError("Type of spice circuit to use must be "+
                                 "specified")

            copy = Signal(signal.times-signal.times[0], signal.values)
            ngspice_in = pyspice.SpiceSignal(copy)

            circuit = None
            # Try to match circuit name in spice_circuits keys
            for key, val in spice_circuits.items():
                if key in self.envelope_method:
                    circuit = val
                    break
            # If circuit not matched, try manual matching of circuit name
            if circuit is None:
                if "simple" in self.envelope_method:
                    circuit = spice_circuits['basic']
                elif ("log amp" in self.envelope_method or
                      "logarithmic amp" in self.envelope_method):
                    circuit = spice_circuits['logamp']
                elif "rectifier" in self.envelope_method:
                    circuit = spice_circuits['bridge']
            # If still no circuits match, raise error
            if circuit is None:
                raise ValueError("Circuit '"+self.envelope_method+
                                 "' not implemented")

            simulator = circuit.simulator(
                temperature=25, nominal_temperature=25,
                ngspice_shared=ngspice_in.shared
            )
            analysis = simulator.transient(step_time=signal.dt,
                                           end_time=copy.times[-1])
            return Signal(signal.times, analysis.output,
                          value_type=signal.value_type)

        else:
            raise ValueError("No envelope method matching '"+
                             self.envelope_method+"'")

    def front_end(self, signal):
        """
        Apply front-end processes to a signal and return the output.

        The front-end consists of the full ARA electronics chain (including
        amplification) and signal clipping, plus an additional amplification
        and envelope circuit.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to apply the front-end processes.

        Returns
        -------
        Signal
            Signal processed by the antenna front end.

        See Also
        --------
        EnvelopeSystem.make_envelope : Return the signal envelope based on the
                                       antenna's ``envelope_method``.
        pyrex.custom.ara.antenna.ARAAntennaSystem.front_end :
            Apply front-end processes to a signal and return the output.

        """
        amplified = super().front_end(signal)
        return self.make_envelope(amplified * self.envelope_amplification)

        # # Two options for downsampling:
        # envelope = self.make_envelope(copy)
        # time = envelope.times[-1] - envelope.times[0]
        # sampling_time = 1e-9
        # npts = time / sampling_time

        # # Option 1
        # downsampled_times = np.linspace(envelope.times[0], envelope.times[-1],
        #                                 num=npts+1)
        # return envelope.with_times()

        # # Option 2
        # envelope.resample(npts)
        # return envelope

    def envelopeless_front_end(self, signal):
        """
        Apply front-end processes to a signal and return the output.

        The front-end consists of the full ARA electronics chain (including
        amplification) and signal clipping. Does not include the extra
        amplification and envelope circuit.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to apply the front-end processes.

        Returns
        -------
        Signal
            Signal processed by the ARA antenna front end.

        See Also
        --------
        pyrex.custom.ara.antenna.ARAAntennaSystem.front_end :
            Apply front-end processes to a signal and return the output.

        """
        return super().front_end(signal)

    def trigger(self, signal):
        """
        Check if the antenna triggers on a given signal.

        Antenna triggers if the voltage waveform exceeds the trigger threshold
        for the required time over threshold.

        Parameters
        ----------
        signal : Signal
            ``Signal`` object on which to test the trigger condition.

        Returns
        -------
        boolean
            Whether or not the antenna triggers on `signal`.

        """
        imax = len(signal.times)
        i = 0
        while i<imax:
            j = i
            while i<imax-1 and signal.values[i]>self.trigger_threshold:
                i += 1
            if i!=j:
                time = signal.times[i]-signal.times[j]
                if time>self.time_over_threshold:
                    return True
            i += 1
        return False



class EnvelopeHpol(EnvelopeSystem):
    """
    ARA Hpol ("quad-slot") antenna system with front-end processing.

    Consists of an ARA Hpol antenna with typical responses, front-end
    electronics, and amplifier clipping, but with an additional amplification
    and envelope circuit applied after all other front-end processing.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float, optional
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    envelope_amplification : float, optional
        Amplification to be applied to the signal after the typical ARA front
        end, before the envelope circuit.
    envelope_method : {('hilbert', 'analytic', 'spice') + ('basic', 'biased',\
                        'doubler', 'bridge', 'log amp')}, optional
        String describing the circuit (and calculation method) to be used for
        envelope calculation. If the string contains "hilbert", the hilbert
        envelope is used. If the string contains "analytic", an analytic form
        is used to calculate the circuit output. If the string contains
        "spice", ``ngspice`` is used to calculate the circuit output. The
        default value "analytic" uses an analytic diode bridge circuit.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    envelope_amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    envelope_method : str
        String describing the circuit (and calculation method) to be used for
        envelope calculation.
    is_hit
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    """
    def __init__(self, name, position, trigger_threshold, time_over_threshold=0,
                 orientation=(0,0,1), amplification=1, amplifier_clipping=1,
                 envelope_amplification=1, envelope_method="analytic",
                 noisy=True, unique_noise_waveforms=10):
        super().__init__(response_data=HPOL_RESPONSE_DATA,
                         name=name, position=position,
                         trigger_threshold=trigger_threshold,
                         time_over_threshold=time_over_threshold,
                         orientation=orientation,
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         envelope_amplification=envelope_amplification,
                         envelope_method=envelope_method,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)


class EnvelopeVpol(EnvelopeSystem):
    """
    ARA Vpol ("bicone" or "birdcage") antenna system with front-end processing.

    Consists of an ARA Vpol antenna with typical responses, front-end
    electronics, and amplifier clipping, but with an additional amplification
    and envelope circuit applied after all other front-end processing.

    Parameters
    ----------
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float, optional
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    orientation : array_like, optional
        Vector direction of the z-axis of the antenna.
    amplification : float, optional
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    amplifier_clipping : float, optional
        Voltage (V) above which the amplified signal is clipped (in positive
        and negative values).
    envelope_amplification : float, optional
        Amplification to be applied to the signal after the typical ARA front
        end, before the envelope circuit.
    envelope_method : {('hilbert', 'analytic', 'spice') + ('basic', 'biased',\
                        'doubler', 'bridge', 'log amp')}, optional
        String describing the circuit (and calculation method) to be used for
        envelope calculation. If the string contains "hilbert", the hilbert
        envelope is used. If the string contains "analytic", an analytic form
        is used to calculate the circuit output. If the string contains
        "spice", ``ngspice`` is used to calculate the circuit output. The
        default value "analytic" uses an analytic diode bridge circuit.
    noisy : boolean, optional
        Whether or not the antenna should add noise to incoming signals.
    unique_noise_waveforms : int, optional
        The number of expected noise waveforms needed for each received signal
        to have its own noise.

    Attributes
    ----------
    antenna : Antenna
        ``Antenna`` object extended by the front end.
    name : str
        Name of the antenna.
    position : array_like
        Vector position of the antenna.
    trigger_threshold : float
        Threshold (V) for trigger condition. Antenna triggers if the voltage
        value of the waveform exceeds this value.
    time_over_threshold : float
        Time (s) that the voltage waveform must exceed `trigger_threshold` for
        the antenna to trigger.
    envelope_amplification : float
        Amplification to be applied to the signal pre-clipping. Note that the
        usual ARA electronics amplification is already applied without this.
    envelope_method : str
        String describing the circuit (and calculation method) to be used for
        envelope calculation.
    is_hit
    is_hit_mc_truth
    signals
    waveforms
    all_waveforms

    """
    def __init__(self, name, position, trigger_threshold, time_over_threshold=0,
                 orientation=(0,0,1), amplification=1, amplifier_clipping=1,
                 envelope_amplification=1, envelope_method="analytic",
                 noisy=True, unique_noise_waveforms=10):
        super().__init__(response_data=VPOL_RESPONSE_DATA,
                         name=name, position=position,
                         trigger_threshold=trigger_threshold,
                         time_over_threshold=time_over_threshold,
                         orientation=orientation,
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         envelope_amplification=envelope_amplification,
                         envelope_method=envelope_method,
                         noisy=noisy,
                         unique_noise_waveforms=unique_noise_waveforms)
