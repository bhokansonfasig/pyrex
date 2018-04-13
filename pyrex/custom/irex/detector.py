"""Module containing customized detector geometry classes for IREX"""

import numpy as np
from pyrex.detector import Detector
from .antenna import IREXAntennaSystem


class IREXDetector(Detector):
    """Base class for IREX detector classes which implements the build_antennas
    method, but not set_positions."""
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
                IREXAntennaSystem(name="IREX antenna", position=pos,
                                  trigger_threshold=trigger_threshold,
                                  time_over_threshold=time_over_threshold,
                                  amplification=amplification,
                                  orientation=(0,0,1), noisy=noisy,
                                  envelope_method=envelope_method)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))
            ant.antenna.set_orientation(*orientation_scheme(i, ant))



class IREXGrid(IREXDetector):
    """Class for (semi)automatically generating a rectangular grid of strings
    of antennas, which can then be iterated over."""
    def set_positions(self, number_of_strings=1, string_separation=500,
                      antennas_per_string=2, antenna_separation=40,
                      lowest_antenna=-200):
        """Generates antenna positions in a grid of strings.
        Takes as arguments the number of strings, the distance between strings,
        the number of antennas per string, the separation (in z) of the
        antennas on the string, and the position of the lowest antenna."""
        self.antenna_positions = []
        n_x = int(np.sqrt(number_of_strings))
        n_y = int(number_of_strings/n_x)
        n_z = antennas_per_string
        dx = string_separation
        dy = string_separation
        dz = antenna_separation
        for i in range(n_x):
            x = -dx*n_x/2 + dx/2 + dx*i
            for j in range(n_y):
                y = -dy*n_y/2 + dy/2 + dy*j
                for k in range(n_z):
                    z = lowest_antenna + dz*k
                    self.antenna_positions.append((x,y,z))


class IREXClusteredGrid(IREXDetector):
    """Class for (semi)automatically generating a rectangular grid of clusters
    of strings of antennas, which can then be iterated over."""
    def set_positions(self, number_of_stations=1, station_separation=500,
                      antennas_per_string=2, antenna_separation=40,
                      lowest_antenna=-200, strings_per_station=4,
                      string_separation=50):
        """Generates antenna positions in a grid of strings.
        Takes as arguments the number of stations, the distance between
        stations, the number of antennas per string, the separation (in z) of the
        antennas on the string, the position of the lowest antenna, and the name
        of the geometry to use. Optional parameters (depending on the geometry)
        are the number of strings per station and the distance from station to
        string."""
        self.antenna_positions = []
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


class IREXCoxeterClusters(IREXDetector):
    """Class for (semi)automatically generating a rectangular grid of 
    Coxeter-plane-like clusters (one string at center) of strings of antennas,
    which can then be iterated over."""
    def set_positions(self, number_of_stations=1, station_separation=500,
                      antennas_per_string=2, antenna_separation=40,
                      lowest_antenna=-200, strings_per_station=4,
                      string_separation=25):
        """Generates antenna positions in a grid of strings.
        Takes as arguments the number of stations, the distance between
        stations, the number of antennas per string, the separation (in z) of the
        antennas on the string, the position of the lowest antenna, and the name
        of the geometry to use. Optional parameters (depending on the geometry)
        are the number of strings per station and the distance from station to
        string."""
        self.antenna_positions = []
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
                    if L==0:
                        x = x_st
                        y = y_st
                    else:
                        angle = 0 if L==1 else 2*np.pi * (L-1)/(n_r-1)
                        x = x_st + dr*np.cos(angle)
                        y = y_st + dr*np.sin(angle)
                    for k in range(n_z):
                        z = lowest_antenna + dz*k
                        self.antenna_positions.append((x,y,z))


class IREXPairedGrid(IREXDetector):
    """Class for (semi)automatically generating a rectangular grid of strings
    of antennas, which can then be iterated over."""
    def set_positions(self, number_of_strings=1, string_separation=500,
                      antennas_per_string=16, antenna_separation=10,
                      lowest_antenna=-95, antennas_per_clump=2,
                      clump_separation=1):
        """Generates antenna positions in a grid of strings.
        Takes as arguments the number of strings, the distance between strings,
        the number of antennas per string, the separation (in z) of the
        antennas on the string, and the position of the lowest antenna."""
        self.antenna_positions = []
        n_x = int(np.sqrt(number_of_strings))
        n_y = int(number_of_strings/n_x)
        n_z = antennas_per_string
        dx = string_separation
        dy = string_separation
        dz = antenna_separation
        for i in range(n_x):
            x = -dx*n_x/2 + dx/2 + dx*i
            for j in range(n_y):
                y = -dy*n_y/2 + dy/2 + dy*j
                for k in range(int(n_z/antennas_per_clump)):
                    z = lowest_antenna + dz*k
                    for L in range(antennas_per_clump):
                        self.antenna_positions.append((x, y,
                                                       z+L*clump_separation))
