"""Module containing customized antenna classes for IREX"""

import numpy as np
import scipy.signal
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel

from .frontends import pyspice, spice_circuits, basic_envelope_model

class IREXBaseAntenna(Antenna):
    """Antenna to be used in IREXAntenna class. Has a position (m),
    center frequency (Hz), bandwidth (Hz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), effective_height=None, noisy=True):
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
                 orientation=(0,0,1), amplification=1, amplifier_clipping=3,
                 noisy=True, envelope_method="analytic"):
        self.name = str(name)
        self.position = position
        self.change_antenna(orientation=orientation, noisy=noisy)

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.trigger_threshold = trigger_threshold
        self.time_over_threshold = time_over_threshold

        self.envelope_method = envelope_method

        self._signals = []
        self._all_waveforms = []
        self._triggers = []

    def change_antenna(self, center_frequency=250e6, bandwidth=300e6,
                       resistance=100, orientation=(0,0,1),
                       effective_height=None, noisy=True):
        """Changes attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        self.antenna = IREXBaseAntenna(position=self.position,
                                       center_frequency=center_frequency,
                                       bandwidth=bandwidth,
                                       resistance=resistance,
                                       orientation=orientation,
                                       effective_height=effective_height,
                                       noisy=noisy)

    def make_envelope(self, signal):
        """Return the signal envelope based on the antenna's envelope_method."""
        if "hilbert" in self.envelope_method:
            return Signal(signal.times, signal.envelope,
                          value_type=signal.value_type)

        elif "analytic" in self.envelope_method:
            if ("basic" in self.envelope_method or
                    self.envelope_method=="analytic"):
                return basic_envelope_model(signal)
            else:
                raise ValueError("Only basic envelope circuit is modeled analytically")

        elif "spice" in self.envelope_method:
            if not(pyspice.__available__):
                raise ModuleNotFoundError(pyspice.__modulenotfound__)

            if self.envelope_method=="spice":
                raise ValueError("Type of spice circuit to use must be specified")

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
                raise ValueError("Circuit '"+self.envelope_method+"' not implemented")

            simulator = circuit.simulator(
                temperature=25, nominal_temperature=25,
                ngspice_shared=ngspice_in.shared
            )
            analysis = simulator.transient(step_time=signal.dt,
                                           end_time=copy.times[-1])
            return Signal(signal.times, analysis.output,
                          value_type=signal.value_type)

        else:
            raise ValueError("No envelope method matching '"+self.envelope_method+"'")

    def front_end_processing(self, signal):
        """Apply the front-end processing of the antenna signal, including
        amplification, clipping, and envelope processing."""
        amplified_values = np.clip(signal.values*self.amplification,
                                   a_min=-self.amplifier_clipping,
                                   a_max=self.amplifier_clipping)
        copy = Signal(signal.times, amplified_values)
        return self.make_envelope(copy)

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

    @property
    def is_hit(self):
        return len(self.waveforms)>0

    def is_hit_during(self, times):
        return self.trigger(self.full_waveform(times))

    @property
    def signals(self):
        # Process any unprocessed antenna signals
        while len(self._signals)<len(self.antenna.signals):
            signal = self.antenna.signals[len(self._signals)]
            self._signals.append(self.front_end_processing(signal))
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
        # Process any unprocessed antenna waveforms
        while len(self._all_waveforms)<len(self.antenna.signals):
            signal = self.antenna.signals[len(self._all_waveforms)]
            t = signal.times
            long_times = np.concatenate((t-t[-1]+t[0], t[1:]))
            long_signal = signal.with_times(long_times)
            long_noise = self.antenna.make_noise(long_times)
            long_waveform = self.front_end_processing(long_signal+long_noise)
            self._all_waveforms.append(long_waveform.with_times(t))
        # Return envelopes of antenna waveforms
        return self._all_waveforms

    def full_waveform(self, times):
        # Process full antenna waveform
        # TODO: Optimize this so it doesn't have to double the amount of time
        # And same for the similar method above in all_waveforms
        long_times = np.concatenate((times-times[-1]+times[0], times[1:]))
        preprocessed = self.antenna.full_waveform(long_times)
        long_waveform = self.front_end_processing(preprocessed)
        return long_waveform.with_times(times)

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
                       amplification=1,
                       naming_scheme=lambda i, ant: "ant_"+str(i),
                       orientation_scheme=lambda i, ant: ((0,0,1), (1,0,0)),
                       noisy=True, envelope_method="analytic"):
        """Sets up IREXAntennas at the positions stored in the class.
        Takes as arguments the trigger threshold, optional time over
        threshold, and whether to add noise to the waveforms.
        Other optional arguments include a naming scheme and orientation scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the orientation scheme
        should return the orientation z-axis and x-axis of the antenna."""
        self.antennas = []
        for pos in self.antenna_positions:
            self.antennas.append(
                IREXAntenna(name="IREX antenna", position=pos,
                            trigger_threshold=trigger_threshold,
                            time_over_threshold=time_over_threshold,
                            amplification=amplification,
                            orientation=(0,0,1), noisy=noisy,
                            envelope_method=envelope_method)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))
            ant.antenna.set_orientation(*orientation_scheme(i, ant))

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
