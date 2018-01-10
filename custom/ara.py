"""Module containing customized classes for ARA (Askaryan Radio Array)"""

import os.path
import numpy as np
import scipy.signal

from pyrex.internal_functions import normalize
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel



def read_response_data(filename):
    """Gather antenna response data from a data file. Returns the data as a
    dictionary with keys (freq, theta, phi) and values (gain, phase).
    Also returns a set of the frequencies appearing in the keys."""
    data = {}
    freqs = set()
    freq = 0
    with open(filename) as f:
        for line in f:
            words = line.split()
            if line.startswith('freq'):
                freq = 1
                if words[-1]=="Hz":
                    pass
                elif words[-1]=="kHz":
                    freq *= 1e3
                elif words[-1]=="MHz":
                    freq *= 1e6
                elif words[-1]=="GHz":
                    freq *= 1e9
                else:
                    raise ValueError("Cannot parse line: '"+line+"'")
                freq *= float(words[-2])
                freqs.add(freq)
            elif len(words)==5 and words[0]!="Theta":
                theta = int(words[0])
                phi = int(words[1])
                db_gain = float(words[2])
                gain = float(words[3])
                phase = float(words[4])
                data[(freq, theta, phi)] = (gain, phase)

    return data, freqs


ARA_DATA_DIR = os.path.dirname(__file__)
VPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_bicone6in_output.txt")
HPOL_DATA_FILE = os.path.join(ARA_DATA_DIR, "ARA_dipoletest1_output.txt")
VPOL_RESPONSE, VPOL_FREQS = read_response_data(VPOL_DATA_FILE)
HPOL_RESPONSE, HPOL_FREQS = read_response_data(HPOL_DATA_FILE)


class ARABaseAntenna(Antenna):
    """Antenna to be used in ARAAntenna class. Has a position (m),
    center frequency (Hz), bandwidth (Hz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 orientation=(0,0,1), effective_height=None,
                 response_data=None, response_freqs=None, noisy=True):
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

        self._response_data = response_data
        self._response_freqs = response_freqs
        if self._response_freqs is None and self._response_data is not None:
            self._response_freqs = set()
            for key in self._response_data:
                self._response_freqs.add(key[0])


    def polarization_gain(self, polarization):
        """Polarization gain is simply the dot product of the polarization
        with the antenna's z-axis."""
        return np.vdot(self.z_axis, polarization)


    def generate_freq_gains(self, theta, phi):
        """Generate arrays of frequencies and gains for given angles."""
        if self._response_data is None:
            return np.array([1]), np.array([1])

        theta_under = 5*int(theta/5)
        theta_over = 5*(int(theta/5)+1)
        phi_under = 5*int(phi/5)
        phi_over = 5*(int(phi/5)+1)
        t = (theta - theta_under) / (theta_over - theta_under)
        u = (phi - phi_under) / (phi_over - phi_under)

        theta_over %= 180
        phi_over %= 360

        nfreqs = len(self._response_freqs)
        gain_ij = np.zeros(nfreqs)
        gain_i1j = np.zeros(nfreqs)
        gain_ij1 = np.zeros(nfreqs)
        gain_i1j1 = np.zeros(nfreqs)
        for f, freq in enumerate(self._response_freqs):
            # TODO: Implement phase shift as imaginary part of gain
            gain_ij[f] = self._response_data[(freq, theta_under, phi_under)][0]
            gain_i1j[f] = self._response_data[(freq, theta_over, phi_under)][0]
            gain_ij1[f] = self._response_data[(freq, theta_under, phi_over)][0]
            gain_i1j1[f] = self._response_data[(freq, theta_over, phi_over)][0]

        freqs = np.array(list(self._response_freqs))
        gains = ((1-t)*(1-u)*gain_ij + t*(1-u)*gain_i1j +
                 (1-t)*u*gain_ij1 + t*u*gain_i1j1)

        return freqs, gains


    def receive(self, signal, origin=None, polarization=None):
        """Process incoming signal according to the filter function and
        store it to the signals list. Subclasses may extend this fuction,
        but should end with super().receive(signal)."""
        copy = Signal(signal.times, signal.values, value_type=Signal.ValueTypes.voltage)
        copy.filter_frequencies(self.response)

        if origin is not None:
            # Calculate theta and phi relative to the orientation
            r, theta, phi = self._convert_to_antenna_coordinates(origin)
            freq_data, gain_data = self.generate_freq_gains(theta, phi)
            def interpolated_response(frequencies):
                return np.interp(frequencies, freq_data, gain_data)
            copy.filter_frequencies(interpolated_response)

        if polarization is None:
            p_gain = 1
        else:
            p_gain = self.polarization_gain(normalize(polarization))

        signal_factor = p_gain * self.efficiency

        if signal.value_type==Signal.ValueTypes.voltage:
            pass
        elif signal.value_type==Signal.ValueTypes.field:
            signal_factor /= self.antenna_factor
        else:
            raise ValueError("Signal's value type must be either "
                             +"voltage or field. Given "+str(signal.value_type))

        copy.values *= signal_factor
        self.signals.append(copy)



class ARAAntenna:
    """ARA antenna system consisting of antenna, amplification,
    and tunnel diode response."""
    def __init__(self, name, position, power_threshold, response_data=None,
                 response_freqs=None, orientation=(0,0,1), amplification=1,
                 amplifier_clipping=3, noisy=True):
        self.name = str(name)
        self.position = position
        self.change_antenna(response_data=response_data,
                            response_freqs=response_freqs,
                            orientation=orientation, noisy=noisy)

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.power_threshold = power_threshold

        self._signals = []
        self._all_waveforms = []
        self._triggers = []

    def change_antenna(self, center_frequency=500e6, bandwidth=700e6,
                       resistance=100, orientation=(0,0,1),
                       response_data=None, response_freqs=None,
                       effective_height=None, noisy=True):
        """Changes attributes of the antenna including center frequency (Hz),
        bandwidth (Hz), resistance (ohms), orientation, and effective
        height (m)."""
        self.antenna = ARABaseAntenna(position=self.position,
                                      center_frequency=center_frequency,
                                      bandwidth=bandwidth,
                                      resistance=resistance,
                                      orientation=orientation,
                                      effective_height=effective_height,
                                      response_data=response_data,
                                      response_freqs=response_freqs,
                                      noisy=noisy)

    def tunnel_diode(self, signal):
        """Return the signal response from the tunnel diode."""
        return signal

    def front_end_processing(self, signal):
        """Apply the front-end processing of the antenna signal, including
        amplification, clipping, and envelope processing."""
        amplified_values = np.clip(signal.values*self.amplification,
                                   a_min=-self.amplifier_clipping,
                                   a_max=self.amplifier_clipping)
        copy = Signal(signal.times, amplified_values)
        return self.tunnel_diode(copy)

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
        if self.antenna._noise_master is None:
            self.antenna.make_noise([0,1])
        return (np.max(signal.values) >
                -1 * self.power_threshold * self.antenna._noise_master.rms)



class HpolAntenna(ARAAntenna):
    """ARA Hpol ("quad-slot") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=3, noisy=True):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         response_data=HPOL_RESPONSE,
                         response_freqs=HPOL_FREQS,
                         orientation=(1,0,0),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy)


class VpolAntenna(ARAAntenna):
    """ARA Vpol ("bicone" or "birdcage") antenna system consisting of antenna,
    amplification, and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 amplification=1, amplifier_clipping=3, noisy=True):
        super().__init__(name=name, position=position,
                         power_threshold=power_threshold,
                         response_data=VPOL_RESPONSE,
                         response_freqs=VPOL_FREQS,
                         orientation=(0,0,1),
                         amplification=amplification,
                         amplifier_clipping=amplifier_clipping,
                         noisy=noisy)



def convert_hex_coords(hex_coords, unit=1):
    """Converts from hexagonal coordinate system to x, y coordinates.
    Optional unit will multiply the x, y coordinate result."""
    x = (hex_coords[0] - hex_coords[1]/2) * unit
    y = (hex_coords[1] * np.sqrt(3)/2) * unit
    return (x, y)



class ARADetector:
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
    def __init__(self, number_of_stations=1, station_separation=2000,
                 antennas_per_string=4, antenna_separation=10,
                 lowest_antenna=-200, strings_per_station=4,
                 string_separation=10):
        self.antenna_positions = []

        # Set positions of stations in hexagonal spiral
        if number_of_stations<=0:
            raise ValueError("Detector has no stations")
        station_positions = [(0, 0)]
        per_side = 1
        per_ring = 1
        ring_count = 0
        hex_pos = (0, 0)
        while len(station_positions)<number_of_stations:
            ring_count += 1
            if ring_count==per_ring:
                per_side += 1
                per_ring = (per_side-1)*6
                ring_count = 0
                hex_pos = (hex_pos[0]+0, hex_pos[1]-1)

            side = int(ring_count/per_ring*6)
            if side==0:
                hex_pos = (hex_pos[0]+1, hex_pos[1]+1)
            elif side==1:
                hex_pos = (hex_pos[0],   hex_pos[1]+1)
            elif side==2:
                hex_pos = (hex_pos[0]-1, hex_pos[1])
            elif side==3:
                hex_pos = (hex_pos[0]-1, hex_pos[1]-1)
            elif side==4:
                hex_pos = (hex_pos[0],   hex_pos[1]-1)
            elif side==5:
                hex_pos = (hex_pos[0]+1, hex_pos[1])

            station_positions.append(
                convert_hex_coords(hex_pos, unit=station_separation)
            )

        # Set antennas at each station
        for stat_index, base_pos in enumerate(station_positions):
            for str_index in range(strings_per_station):
                angle = str_index/strings_per_station * 2*np.pi
                x = base_pos[0] + string_separation*np.cos(angle)
                y = base_pos[1] + string_separation*np.sin(angle)
                for ant_index in range(antennas_per_string):
                    z = lowest_antenna + ant_index*antenna_separation
                    self.antenna_positions.append((x,y,z))

        for pos in self.antenna_positions:
            if pos[2]>0:
                raise ValueError("Antenna placed outside of ice will cause "
                                 +"unexpected issues")

        self.antennas = []

    def build_antennas(self, power_threshold, amplification=1,
                       naming_scheme=lambda i, ant: "ant_"+str(i),
                       class_scheme=lambda i: HpolAntenna if i%2 else VpolAntenna,
                       noisy=True):
        """Sets up ARAAntennas at the positions stored in the class.
        Takes as arguments the power threshold, amplification, and whether to
        add noise to the waveforms.
        Other optional arguments include a naming scheme and orientation scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the orientation scheme
        should return the orientation z-axis and x-axis of the antenna."""
        self.antennas = []
        for i, pos in enumerate(self.antenna_positions):
            AntennaClass = class_scheme(i)
            self.antennas.append(
                AntennaClass(name="ARA antenna", position=pos,
                             power_threshold=power_threshold,
                             amplification=amplification,
                             noisy=noisy)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))

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
