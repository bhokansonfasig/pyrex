"""Module containing customized classes for IREX"""

import os, os.path
import numpy as np
import scipy.signal
from scipy.special import lambertw
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel

USE_PYSPICE = False

if USE_PYSPICE:
    from PySpice.Spice.NgSpice.Shared import NgSpiceShared
    from PySpice.Spice.Netlist import Circuit
    from PySpice.Spice.Library import SpiceLibrary
    from PySpice.Unit import *


    SPICE_LIBRARY = SpiceLibrary(os.path.join(os.getcwd(), 'spice_models'))

    ENVELOPE_CIRCUIT_C1 = 220@u_pF
    ENVELOPE_CIRCUIT_R1 = 50@u_Ohm

    ENVELOPE_CIRCUIT_C2 = 10@u_nF
    ENVELOPE_CIRCUIT_R2 = 1@u_kOhm
    ENVELOPE_CIRCUIT_R3 = 1@u_kOhm
    ENVELOPE_CIRCUIT_VBIAS = 5@u_V

    ENVELOPE_CIRCUIT = Circuit('Biased Envelope Circuit')
    ENVELOPE_CIRCUIT.include(SPICE_LIBRARY['hsms'])

    ENVELOPE_CIRCUIT.V('in', 'input', ENVELOPE_CIRCUIT.gnd, 'dc 0 external')
    # bias portion
    ENVELOPE_CIRCUIT.C(2, 'input', 1, ENVELOPE_CIRCUIT_C2)
    ENVELOPE_CIRCUIT.R(2, 1, 2, ENVELOPE_CIRCUIT_R2)
    ENVELOPE_CIRCUIT.X('D2', 'hsms', 2, ENVELOPE_CIRCUIT.gnd)
    ENVELOPE_CIRCUIT.R(3, 2, 'bias', ENVELOPE_CIRCUIT_R3)
    ENVELOPE_CIRCUIT.V('bias', 'bias', ENVELOPE_CIRCUIT.gnd, ENVELOPE_CIRCUIT_VBIAS)
    # envelope portion
    ENVELOPE_CIRCUIT.X('D1', 'hsms', 1, 'output')
    ENVELOPE_CIRCUIT.C(1, 'output', ENVELOPE_CIRCUIT.gnd, ENVELOPE_CIRCUIT_C1)
    ENVELOPE_CIRCUIT.R(1, 'output', ENVELOPE_CIRCUIT.gnd, ENVELOPE_CIRCUIT_R1)


    class NgSpiceSharedSignal(NgSpiceShared):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._signal = None

        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
            voltage[0] = np.interp(time, self._signal.times, self._signal.values)
            return 0

    NGSPICE_SHARED_MASTER = NgSpiceSharedSignal()

    class SpiceSignal:
        def __init__(self, signal, shared=NGSPICE_SHARED_MASTER):
            self.shared = shared
            self.shared._signal = signal


def envelope_model(signal, cap=220e-12, res=50):
    """Model of a basic diode-capacitor-resistor envelope circuit. Takes a
    signal object as the input voltage and returns the output voltage signal
    object."""
    v_c = 0
    v_out = []

    r_d = 25
    i_s = 3e-6
    n = 1.06
    v_t = 26e-3

    # Terms which can be calculated ahead of time to save time in the loop
    charge_exp = np.exp(-signal.dt/(res*cap))
    discharge = i_s*res*(1-charge_exp)
    lambert_factor = n*v_t*res/r_d*(1-charge_exp)
    frac = i_s*r_d/n/v_t
    lambert_exponent = np.log(frac) + frac

    for v_in in signal.values:
        # Calculate exponent of exponential in lambert function instead of
        # calculating exponential directly to avoid overflows
        a = lambert_exponent + (v_in - v_c)/n/v_t
        if a>100:
            # If exponential in lambert function is large enough,
            # use approximation of lambert function
            # (doesn't save time, but does prevent overflows)
            b = np.log(a)
            lambert_term = a - b + b/a
        else:
            # Otherwise, use the lambert function directly
            lambert_term = np.real(lambertw(np.exp(a)))
            if np.isnan(lambert_term):
                # Only seems to happen when np.exp(a) is very close to zero
                # (so lambert_term will also be very close to zero)
                lambert_term = 0

        # Calculate voltage across capacitor after time dt
        v_c = v_c*charge_exp - discharge + lambert_factor*lambert_term
        v_out.append(v_c)

    return Signal(signal.times, v_out, value_type=Signal.ValueTypes.voltage)


class IREXBaseAntenna(Antenna):
    """Antenna to be used in IREXAntenna class. Has a position (m),
    center frequency (Hz), bandwidth (Hz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), effective_height=None,
                 amplification=1, noisy=True):
        if effective_height is None:
            # Calculate length of half-wave dipole
            self.effective_height = 3e8 / center_frequency / 2
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
        # Note: ortho is not normalized, but will be normalized by Antenna's init

        super().__init__(position=position, z_axis=orientation, x_axis=ortho,
                         antenna_factor=1/self.effective_height,
                         efficiency=amplification,
                         temperature=IceModel.temperature(position[2]),
                         freq_range=(f_low, f_high), resistance=resistance,
                         noisy=noisy)

        # Build scipy butterworth filter to speed up response function
        b, a  = scipy.signal.butter(1, 2*np.pi*np.array(self.freq_range),
                                    btype='bandpass', analog=True)
        self.filter_coeffs = (b, a)

    def response(self, frequencies):
        """Butterworth filter response for the antenna's frequency range."""
        angular_freqs = np.array(frequencies) * 2*np.pi
        w, h = scipy.signal.freqs(self.filter_coeffs[0], self.filter_coeffs[1],
                                  angular_freqs)
        return h

    def directional_gain(self, theta, phi):
        """Power gain of dipole antenna goes as sin(theta)^2, so electric field
        gain goes as sin(theta)."""
        return np.sin(theta)

    def polarization_gain(self, polarization):
        """Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis."""
        return np.vdot(self.z_axis, polarization)



class IREXAntenna:
    """IREX antenna system consisting of dipole antenna, low-noise amplifier,
    optional bandpass filter, and envelope circuit."""
    def __init__(self, name, position, trigger_threshold, time_over_threshold=0,
                 orientation=(0,0,1), amplification=1, noisy=True,
                 envelope_method="analytic"):
        self.name = str(name)
        self.position = position
        self.change_antenna(orientation=orientation,
                            amplification=amplification, noisy=noisy)

        self.trigger_threshold = trigger_threshold
        self.time_over_threshold = time_over_threshold

        self.envelope_method = envelope_method

        self._signals = []
        self._all_waveforms = []
        self._triggers = []

    def change_antenna(self, center_frequency=250e6, bandwidth=300e6,
                       resistance=100, orientation=(0,0,1),
                       effective_height=None, amplification=1, noisy=True):
        """Changes attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        self.antenna = IREXBaseAntenna(position=self.position,
                                       center_frequency=center_frequency,
                                       bandwidth=bandwidth,
                                       resistance=resistance,
                                       orientation=orientation,
                                       effective_height=effective_height,
                                       amplification=amplification,
                                       noisy=noisy)

    def make_envelope(self, signal):
        if self.envelope_method=="hilbert":
            return Signal(signal.times, signal.envelope,
                          value_type=signal.value_type)
        elif self.envelope_method=="spice":
            if not(USE_PYSPICE):
                raise ModuleNotFoundError("PySpice was not imported")
            ngspice_in = SpiceSignal(signal)
            simulator = ENVELOPE_CIRCUIT.simulator(
                temperature=25, nominal_temperature=25,
                ngspice_shared=ngspice_in.shared
            )
            analysis = simulator.transient(step_time=signal.dt,
                                           start_time=signal.times[0],
                                           end_time=signal.times[-1])
            return Signal(signal.times, analysis.output,
                          value_type=signal.value_type)
        elif self.envelope_method=="analytic":
            return envelope_model(signal)

    @property
    def is_hit(self):
        return len(self.waveforms)>0

    def is_hit_during(self, times):
        return self.trigger(self.full_waveform(times))

    @property
    def signals(self):
        # Process envelopes of any unprocessed antenna signals
        while len(self._signals)<len(self.antenna.signals):
            signal = self.antenna.signals[len(self._signals)]
            self._signals.append(self.make_envelope(signal))
        # Return envelopes of antenna signals
        return self._signals

    @property
    def waveforms(self):
        # Process any unprocessed triggers
        all_waves = self.all_waveforms
        while len(self._triggers)<len(all_waves):
            waveform = all_waves[len(self._triggers)]
            self._triggers.append(self.trigger(waveform))

        return [wave for wave, triggered in zip(all_waves, self._triggers)
                if triggered]

    @property
    def all_waveforms(self):
        # Process envelopes of any unprocessed antenna waveforms
        while len(self._all_waveforms)<len(self.antenna.all_waveforms):
            signal = self.antenna.all_waveforms[len(self._all_waveforms)]
            self._all_waveforms.append(self.make_envelope(signal))
        # Return envelopes of antenna waveforms
        return self._all_waveforms

    def full_waveform(self, times):
        preprocessed = self.antenna.full_waveform(times)
        return self.make_envelope(preprocessed)

    def receive(self, signal, origin=None, polarization=None):
        return self.antenna.receive(signal, origin=origin,
                                    polarization=polarization)

    def clear(self):
        self._signals.clear()
        self._all_waveforms.clear()
        self._triggers.clear()
        self.antenna.clear()

    def trigger(self, signal):
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



class IREXDetector:
    """Class for automatically generating antenna positions based on geometry
    criteria. Takes as arguments the number of stations, the distance between
    stations, the number of antennas per string, the separation (in z) of the
    antennas on the string, the position of the lowest antenna, and the name
    of the geometry to use. Optional parameters (depending on the geometry)
    are the number of strings per station and the distance from station to
    string.
    The build_antennas method is responsible for actually placing antennas
    at the generated positions, after which the class can be directly iterated
    to iterate over the antennas."""
    def __init__(self, number_of_stations, station_separation,
                 antennas_per_string, antenna_separation, lowest_antenna,
                 geometry="grid", strings_per_station=1, string_separation=100):
        self.antenna_positions = []

        if "grid" in geometry.lower():
            n_x = int(np.sqrt(number_of_stations))
            n_y = int(number_of_stations/n_x)
            n_z = antennas_per_string
            dx = station_separation
            dy = station_separation
            dz = antenna_separation
            for i in range(n_x):
                x = -dx*n_x/2 + dx/2 + dx*i
                for j in range(n_y):
                    y = -dy*n_y/2 + dy/2 + dy*j
                    for k in range(n_z):
                        z = lowest_antenna + dz*k
                        self.antenna_positions.append((x,y,z))

        elif "cluster" in geometry.lower():
            n_x = int(np.sqrt(number_of_stations))
            n_y = int(number_of_stations/n_x)
            n_z = antennas_per_string
            n_r = strings_per_station
            dx = station_separation
            dy = station_separation
            dz = antenna_separation
            dr = string_separation
            for i in range(n_x):
                x_st = -dx*n_x/2 + dx/2 + dx*i
                for j in range(n_y):
                    y_st = -dy*n_y/2 + dy/2 + dy*j
                    for L in range(n_r):
                        angle = 2*np.pi * L/n_r
                        x = x_st + dr*np.cos(angle)
                        y = y_st + dr*np.sin(angle)
                        for k in range(n_z):
                            z = lowest_antenna + dz*k
                            self.antenna_positions.append((x,y,z))

        for pos in self.antenna_positions:
            if pos[2]>0:
                raise ValueError("Antenna placed outside of ice will cause "
                                 +"unexpected issues")

        self.antennas = []

    def build_antennas(self, trigger_threshold, time_over_threshold=0,
                       naming_scheme=lambda i, ant: "ant_"+str(i),
                       polarization_scheme=lambda i, ant: (0,0,1), noisy=True,
                       envelope_method="analytic"):
        """Sets up IREXAntennas at the positions stored in the class.
        Takes as arguments the trigger threshold, optional time over
        threshold, and whether to add noise to the waveforms.
        Other optional arguments include a naming scheme and polarization scheme
        which are functions taking the antenna index i and the antenna object
        and should return the name and polarization of the antenna,
        respectively."""
        self.antennas = []
        for pos in self.antenna_positions:
            self.antennas.append(
                IREXAntenna(name="IREX antenna", position=pos,
                            trigger_threshold=trigger_threshold,
                            time_over_threshold=time_over_threshold,
                            orientation=(0,0,1), noisy=noisy,
                            envelope_method=envelope_method)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))
            ant.polarization = polarization_scheme(i, ant)

    def __iter__(self):
        self._iter_counter = 0
        self._iter_max = len(self.antennas)
        return self

    def __next__(self):
        self._iter_counter += 1
        if self._iter_counter > self._iter_max:
            raise StopIteration
        else:
            return self.antennas[self._iter_counter-1]

    def __len__(self):
        return len(self.antennas)

    def __getitem__(self, key):
        return self.antennas[key]

    def __setitem__(self, key, value):
        self.antennas[key] = value
