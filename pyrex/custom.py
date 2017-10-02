"""Module containing customized classes for IREX"""

import numpy as np
import scipy.signal
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel

class IREXBaseAntenna(Antenna):
    """Antenna to be used in IREXAntenna class. Has a position (m),
    center frequency (MHz), bandwidth (MHz), resistance (ohm),
    effective height (m), and polarization direction."""
    def __init__(self, position, center_frequency, bandwidth, resistance,
                 effective_height, polarization=(0,0,1), noisy=True):
        # Get the critical frequencies in Hz
        f_low = (center_frequency - bandwidth/2) * 1e6
        f_high = (center_frequency + bandwidth/2) * 1e6
        super().__init__(position=position,
                         temperature=IceModel.temperature(position[2]),
                         freq_range=(f_low, f_high), resistance=resistance,
                         noisy=noisy)
        self.effective_height = effective_height
        self.polarization = (np.array(polarization)
                             / np.linalg.norm(polarization))

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

    def receive(self, signal, polarization=[0,0,1]):
        """Apply polarization effect to signal, then proceed with usual
        antenna reception."""
        scaled_values = (signal.values * self.effective_height
                         * np.abs(np.vdot(self.polarization, polarization)))
        super().receive(Signal(signal.times, scaled_values))


class IREXAntenna:
    """IREX antenna system consisting of dipole antenna, low-noise amplifier,
    optional bandpass filter, and envelope circuit."""
    def __init__(self, name, position, trigger_threshold, time_over_threshold=0,
                 polarization=(0,0,1), noisy=True):
        self.name = str(name)
        self.position = position
        self.change_antenna(polarization=polarization, noisy=noisy)

        self.trigger_threshold = trigger_threshold
        self.time_over_threshold = time_over_threshold

        self._triggers = []

    def change_antenna(self, center_frequency=250, bandwidth=300,
                       resistance=100, polarization=(0,0,1), noisy=True):
        """Changes attributes of the antenna including center frequency (MHz),
        bandwidth (MHz), and resistance (ohms)."""
        h = 3e8 / (center_frequency*1e6) / 2
        self.antenna = IREXBaseAntenna(position=self.position,
                                       center_frequency=center_frequency,
                                       bandwidth=bandwidth,
                                       resistance=resistance,
                                       effective_height=h,
                                       polarization=polarization, noisy=noisy)

    @property
    def is_hit(self):
        return len(self.waveforms)>0

    @property
    def signals(self):
        # Return envelopes of antenna signals
        return [Signal(s.times, s.envelope) for s in self.antenna.signals]

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
        # Return envelopes of antenna waveforms
        return [Signal(w.times, w.envelope) for w in self.antenna.all_waveforms]

    def receive(self, signal, polarization=[0,0,1]):
        return self.antenna.receive(signal, polarization=polarization)

    def clear(self):
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
            n_x = int(number_of_stations/2)
            n_y = int(number_of_stations - n_x)
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

        self.antennas = []

    def build_antennas(self, trigger_threshold, time_over_threshold=0,
                       naming_scheme=lambda i, ant: "ant_"+str(i),
                       polarization_scheme=lambda i, ant: (0,0,1), noisy=True):
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
                            polarization=(0,0,1), noisy=noisy)
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
