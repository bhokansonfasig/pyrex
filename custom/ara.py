"""Module containing customized classes for ARA (Askaryan Radio Array)"""

import numpy as np

from pyrex.signals import Signal
from pyrex.antenna import Antenna


class ARABaseAntenna(Antenna):
    """Antenna to be used in ARAAntenna class. Has a position (m),
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



class ARAAntenna:
    """ARA antenna system consisting of dipole antenna, amplification,
    and tunnel diode response."""
    def __init__(self, name, position, power_threshold,
                 orientation=(0,0,1), amplification=1, amplifier_clipping=3,
                 noisy=True):
        self.name = str(name)
        self.position = position
        self.change_antenna(orientation=orientation, noisy=noisy)

        self.amplification = amplification
        self.amplifier_clipping = amplifier_clipping

        self.power_threshold = power_threshold

        self._signals = []
        self._all_waveforms = []
        self._triggers = []

    def change_antenna(self, center_frequency=500e6, bandwidth=700e6,
                       resistance=100, orientation=(0,0,1),
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
        return (np.max(signal.values)**2/self.antenna.resistance
                > self.power_threshold)



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
                       orientation_scheme=lambda i, ant: (0,0,1), noisy=True):
        """Sets up ARAAntennas at the positions stored in the class.
        Takes as arguments the trigger threshold, optional time over
        threshold, and whether to add noise to the waveforms.
        Other optional arguments include a naming scheme and orientation scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the orientation scheme
        should return the orientation z-axis and x-axis of the antenna."""
        self.antennas = []
        for pos in self.antenna_positions:
            self.antennas.append(
                ARAAntenna(name="ARA antenna", position=pos,
                           power_threshold=power_threshold,
                           amplification=amplification,
                           orientation=(0,0,1), noisy=noisy)
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
