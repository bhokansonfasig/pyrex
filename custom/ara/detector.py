"""Module containing customized detector geometry classes for ARA"""

import numpy as np
from pyrex.signals import Signal
from pyrex.detector import Detector
from pyrex.ice_model import IceModel
from .antenna import HpolAntenna, VpolAntenna


def convert_hex_coords(hex_coords, unit=1):
    """Converts from hexagonal coordinate system to x, y coordinates.
    Optional unit will multiply the x, y coordinate result."""
    x = (hex_coords[0] - hex_coords[1]/2) * unit
    y = (hex_coords[1] * np.sqrt(3)/2) * unit
    return (x, y)



class ARAString(Detector):
    """String of ARA Hpol and Vpol antennas. Sets positions of antennas on
    string based on the given arguments. Sets build_antennas method for setting
    antenna characteristics."""
    def set_positions(self, x, y, antennas_per_string=4,
                      antenna_separation=10, lowest_antenna=-200):
        """Generates antenna positions along the string."""
        if hasattr(antenna_separation, '__len__'):
            if len(antenna_separation)!=antennas_per_string-1:
                raise ValueError("Bad number of antenna separations given")
            def z_sep(i):
                return antenna_separation[i-1]
        else:
            def z_sep(i):
                return antenna_separation
        for i in range(antennas_per_string):
            z = lowest_antenna if i==0 else z+z_sep(i)
            self.antenna_positions.append((x, y, z))

    def build_antennas(self, power_threshold, amplification=1,
                       naming_scheme=lambda i, ant: ant.name[:4]+"_"+str(i),
                       class_scheme=lambda i: HpolAntenna if i%2 else VpolAntenna,
                       noisy=True):
        """Sets up ARA antennas at the positions stored in the class.
        Takes as arguments the power threshold, amplification, and whether to
        add noise to the waveforms.
        Other optional arguments include a naming scheme and class scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the class scheme
        should return HpolAntenna or VpolAntenna for each antenna."""
        for i, pos in enumerate(self.antenna_positions):
            AntennaClass = class_scheme(i)
            self.subsets.append(
                AntennaClass(name=AntennaClass.__name__, position=pos,
                             power_threshold=power_threshold,
                             amplification=amplification,
                             noisy=noisy)
            )
        for i, ant in enumerate(self.subsets):
            ant.name = str(naming_scheme(i, ant))

    def triggered(self, antenna_requirement=1):
        """Test whether the number of hit antennas meets the given antenna
        trigger requirement."""
        antennas_hit = sum(1 for ant in self if ant.is_hit)
        return antennas_hit>=antenna_requirement



class PhasedArrayString(Detector):
    """Phased array string of closely packed ARA antennas. Sets positions of
    antennas on string based on the given arguments. Sets build_antennas method
    for setting antenna characteristics."""
    def set_positions(self, x, y, antennas_per_string=10,
                      antenna_separation=1, lowest_antenna=-100,
                      antenna_type=VpolAntenna):
        """Generates antenna positions along the string."""
        self.antenna_type = antenna_type
        for i in range(antennas_per_string):
            z = lowest_antenna + i*antenna_separation
            self.antenna_positions.append((x, y, z))

    def build_antennas(self, power_threshold, amplification=1,
                       naming_scheme=lambda i, ant: ant.name[:4]+"_"+str(i),
                       class_scheme=lambda i: HpolAntenna if i%2 else VpolAntenna,
                       noisy=True):
        """Sets up ARA antennas at the positions stored in the class.
        Takes as arguments the power threshold, amplification, antenna type,
        and whether to add noise to the waveforms.
        The optional argument is a naming scheme which is a function taking the
        antenna index i and the antenna object and returnint the name for the
        antenna. The class scheme passed does nothing, but is kept so the
        arguments of this function match those of the ARAString class."""
        for i, pos in enumerate(self.antenna_positions):
            self.subsets.append(
                self.antenna_type(name=self.antenna_type.__name__, position=pos,
                                  power_threshold=power_threshold,
                                  amplification=amplification,
                                  noisy=noisy)
            )
        for i, ant in enumerate(self.subsets):
            ant.name = str(naming_scheme(i, ant))

    def triggered(self, beam_threshold, delays=None, angles=None):
        """Test whether the phased array total waveform exceeds the given
        threshold. Delays (in ns) or angles (in degrees) to test can optionally
        be specified, with delays taking precedence."""
        # Explanation of default delays:
        # Delays from 5.94 ns to -5.94 ns cover the edge case of
        # n=1.78 (z=inf) for elevation angles of 90 to -90 degrees.
        # For another edge case of n=1.35 (z=0), only need to cover
        # 4.5 ns to -4.5 ns really, but having extra won't hurt.
        # (All this assuming a z-spacing of 1 m, otherwise multiply
        # by the spacing to get true time bounds)
        # 19 points in the larger range gives a delay resolution of
        # 660 ps (phased array FPGA sampling rate from
        # https://github.com/vPhase/fpga-sim/blob/master/config.py),
        # or equivalently an angular resolution of about 8 or 6 degrees
        # for the edge cases above.
        # I think this is different from the true phased array trigger,
        # which only looks down, so there are 15 positive (or neg?)
        # delays from 0 to 9.24 ns.
        if delays is None:
            dz = self[0].position[2] - self[1].position[2]
            if angles is None:
                # Use default delays described above
                t_max = 5.94e-9 * np.abs(dz)
                delays = np.linspace(-t_max, t_max, 19)
            else:
                # Calculate delays based on elevation angles
                thetas = np.radians(angles)
                n = np.mean([IceModel.index(ant.position[2]) for ant in self])
                v = 3e8 / n
                delays = dz / v * np.sin(thetas)

        rms = sum(ant.antenna._noise_master.rms*ant.amplification
                  for ant in self) / np.sqrt(len(self))

        # Iterate over all waveforms (assume that the antennas are close
        # enough to all see the same rays, i.e. the same index of
        # ant.all_waveforms will be from the same ray for each antenna)
        j = -1
        while True:
            j += 1
            # Center around strongest waveform
            max_i = None
            max_wave = 0
            for i, ant in enumerate(self):
                if (len(ant.all_waveforms)>j and
                        np.max(ant.all_waveforms[j].values**2)>max_wave):
                    max_i = i
                    max_wave = np.max(ant.all_waveforms[j].values**2)
            # Stop waveform iteration once no more waveforms are available
            if max_i is None:
                break
            # Check each delay for trigger
            for delay in delays:
                center_wave = self[max_i].all_waveforms[j]
                total = Signal(center_wave.times, center_wave.values)
                for i, ant in enumerate(self):
                    if i==max_i:
                        continue
                    times = total.times - (max_i-i)*delay
                    add_wave = ant.full_waveform(times)
                    add_wave.times += (max_i-i)*delay
                    total += add_wave.with_times(total.times)
                if np.max(np.abs(total.values))>np.abs(beam_threshold*rms):
                    return True
        return False



class RegularStation(Detector):
    """Station geometry with a number of strings evenly spaced radially around
    the station center. Supports any string type and passes extra keyword
    arguments on to the string class."""
    def set_positions(self, x, y, strings_per_station=4,
                      station_diameter=20, string_type=ARAString,
                      **string_kwargs):
        """Generates string positions around the station."""
        r = station_diameter/2
        for i in range(strings_per_station):
            angle = 2*np.pi * i/strings_per_station
            x_str = x + r*np.cos(angle)
            y_str = y + r*np.sin(angle)
            self.subsets.append(
                string_type(x_str, y_str, **string_kwargs)
            )

    def triggered(self, polarized_antenna_requirement=1):
        """Test whether the number of hit antennas of a single polarization
        meets the given antenna trigger requirement."""
        hpol_hit = sum(1 for ant in self
                       if isinstance(ant, HpolAntenna) and ant.is_hit)
        vpol_hit = sum(1 for ant in self
                       if isinstance(ant, VpolAntenna) and ant.is_hit)
        return (hpol_hit>=polarized_antenna_requirement or
                vpol_hit>=polarized_antenna_requirement)



class AlbrechtStation(Detector):
    """Station geometry proposed by Albrecht with a phased array string of each
    polarization at the station center, plus a number of outrigger strings
    evenly spaced radially around the station center. Supports any string type
    and passes extra keyword arguments on to the outrigger string class."""
    def set_positions(self, x, y, station_diameter=40,
                      hpol_phased_antennas=10, vpol_phased_antennas=10,
                      hpol_phased_separation=1, vpol_phased_separation=1,
                      hpol_phased_lowest=-69, vpol_phased_lowest=-49,
                      outrigger_strings_per_station=3,
                      outrigger_string_type=ARAString,
                      **outrigger_string_kwargs):
        """Generates string positions around the station."""
        # Change defaults for outrigger strings
        if "antennas_per_string" not in outrigger_string_kwargs:
            outrigger_string_kwargs["antennas_per_string"] = 8
        if "antenna_separation" not in outrigger_string_kwargs:
            n = outrigger_string_kwargs["antennas_per_string"]
            sep = [1, 29] * int(n/2)
            outrigger_string_kwargs["antenna_separation"] = sep[:n-1]
        if "lowest_antenna" not in outrigger_string_kwargs:
            outrigger_string_kwargs["lowest_antenna"] = -100

        self.subsets.append(
            PhasedArrayString(x, y, antennas_per_string=hpol_phased_antennas,
                              antenna_separation=hpol_phased_separation,
                              lowest_antenna=hpol_phased_lowest,
                              antenna_type=HpolAntenna)
        )
        self.subsets.append(
            PhasedArrayString(x, y, antennas_per_string=vpol_phased_antennas,
                              antenna_separation=vpol_phased_separation,
                              lowest_antenna=vpol_phased_lowest,
                              antenna_type=VpolAntenna)
        )

        r = station_diameter/2
        for i in range(outrigger_strings_per_station):
            angle = 2*np.pi * i/outrigger_strings_per_station
            x_str = x + r*np.cos(angle)
            y_str = y + r*np.sin(angle)
            self.subsets.append(
                outrigger_string_type(x_str, y_str, **outrigger_string_kwargs)
            )

    def triggered(self, power_threshold):
        """Test whether the phased array strings triggered at some threshold."""
        return (self.subsets[0].triggered(power_threshold=power_threshold) or
                self.subsets[1].triggered(power_threshold=power_threshold))



class HexagonalGrid(Detector):
    """Hexagonal grid of stations or strings, with nearest neighbors
    separated by the given distance. Supports any station or string type and
    passes extra keyword arguments on to the station or string class."""
    def set_positions(self, stations=1, station_separation=2000,
                      station_type=RegularStation, **station_kwargs):
        """Generates hexagonal grid of stations."""
        # Set positions of stations in hexagonal spiral
        if stations<=0:
            raise ValueError("Detector has no stations")
        station_positions = [(0, 0)]
        per_side = 1
        per_ring = 1
        ring_count = 0
        hex_pos = (0, 0)
        while len(station_positions)<stations:
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

        for base_pos in station_positions:
            self.subsets.append(
                station_type(base_pos[0], base_pos[1], **station_kwargs)
            )

    def triggered(self, station_requirement=1, **station_trigger_kwargs):
        """Test whether the number of hit stations meets the given station
        trigger requirement."""
        stations_hit = 0
        for station in self.subsets:
            if station.triggered(**station_trigger_kwargs):
                stations_hit += 1
            if stations_hit>=station_requirement:
                return True
        return stations_hit>=station_requirement
