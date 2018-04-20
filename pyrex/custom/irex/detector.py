"""Module containing customized detector geometry classes for IREX"""

import numpy as np
from pyrex.detector import Detector
from .antenna import IREXAntennaSystem


class IREXString(Detector):
    """String of IREXAntennas. Sets positions of antennas on string based on
    the given arguments. Sets build_antennas method for setting antenna
    characteristics."""
    def set_positions(self, x, y, antennas_per_string=2,
                      antenna_separation=50, lowest_antenna=-100):
        """Generates antenna positions along the string."""
        for i in range(antennas_per_string):
            z = lowest_antenna + i*antenna_separation
            self.antenna_positions.append((x, y, z))

    def build_antennas(self, trigger_threshold, time_over_threshold=0,
                       amplification=1,
                       naming_scheme=lambda i, ant: "ant_"+str(i),
                       orientation_scheme=lambda i, ant: ((0,0,1), (1,0,0)),
                       noisy=True, envelope_method="analytic"):
        """Sets up IREXAntennaSystems at the positions stored in the class.
        Takes as arguments the trigger threshold, optional time over
        threshold, and whether to add noise to the waveforms.
        Other optional arguments include a naming scheme and orientation scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the orientation scheme
        should return the orientation z-axis and x-axis of the antenna."""
        super().build_antennas(antenna_class=IREXAntennaSystem,
                               name="IREX antenna",
                               trigger_threshold=trigger_threshold,
                               time_over_threshold=time_over_threshold,
                               amplification=amplification,
                               orientation=(0, 0, 1), noisy=noisy,
                               envelope_method=envelope_method)
        for i, ant in enumerate(self.subsets):
            ant.name = str(naming_scheme(i, ant))
            ant.antenna.set_orientation(*orientation_scheme(i, ant))

    def triggered(self, antenna_requirement=1):
        """Test whether the number of hit antennas meets the given antenna
        trigger requirement."""
        antennas_hit = sum(1 for ant in self if ant.is_hit)
        return antennas_hit>=antenna_requirement



class RegularStation(Detector):
    """Station geometry with a number of strings evenly spaced radially around
    the station center. Supports any string type and passes extra keyword
    arguments on to the string class."""
    def set_positions(self, x, y, strings_per_station=4,
                      station_diameter=50, string_type=IREXString,
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

    def triggered(self, antenna_requirement=1, string_requirement=1):
        """Test whether the number of hit antennas meets the given antenna
        and string trigger requirements."""
        antennas_hit = sum(1 for ant in self if ant.is_hit)
        strings_hit = sum(1 for string in self.subsets if string.triggered(1))
        return (antennas_hit>=antenna_requirement and
                strings_hit>=string_requirement)



class CoxeterStation(Detector):
    """Station geometry with one string at the station center and the rest of
    the strings evenly spaced radially around the station center. Supports any
    string type and passes extra keyword arguments on to the string class."""
    def set_positions(self, x, y, strings_per_station=4,
                      station_diameter=50, string_type=IREXString,
                      **string_kwargs):
        """Generates string positions around the station."""
        r = station_diameter/2
        for i in range(strings_per_station):
            if i==0:
                x_str = x
                y_str = y
            else:
                angle = 0 if i==1 else 2*np.pi * (i-1)/(strings_per_station-1)
                x_str = x + r*np.cos(angle)
                y_str = y + r*np.sin(angle)
            self.subsets.append(
                string_type(x_str, y_str, **string_kwargs)
            )

    def triggered(self, antenna_requirement=1, string_requirement=1):
        """Test whether the number of hit antennas meets the given antenna
        and string trigger requirements."""
        antennas_hit = sum(1 for ant in self if ant.is_hit)
        strings_hit = sum(1 for string in self.subsets if string.triggered(1))
        return (antennas_hit>=antenna_requirement and
                strings_hit>=string_requirement)



class StationGrid(Detector):
    """Rectangular grid of stations or strings, in a square layout if possible,
    separated by the given distance. Supports any station or string type and
    passes extra keyword arguments on to the station or string class."""
    def set_positions(self, stations=1, station_separation=500,
                      station_type=IREXString, **station_kwargs):
        """Generates rectangular grid of stations."""
        n_x = int(np.sqrt(stations))
        n_y = int(stations/n_x)
        dx = station_separation
        dy = station_separation
        for i in range(n_x):
            x = -dx*n_x/2 + dx/2 + dx*i
            for j in range(n_y):
                y = -dy*n_y/2 + dy/2 + dy*j
                self.subsets.append(
                    station_type(x, y, **station_kwargs)
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
