"""Module containing customized detector geometry classes for ARA"""

import numpy as np
from pyrex.detector import Detector
from .antenna import HpolAntenna, VpolAntenna


def convert_hex_coords(hex_coords, unit=1):
    """Converts from hexagonal coordinate system to x, y coordinates.
    Optional unit will multiply the x, y coordinate result."""
    x = (hex_coords[0] - hex_coords[1]/2) * unit
    y = (hex_coords[1] * np.sqrt(3)/2) * unit
    return (x, y)



class ARADetector(Detector):
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
    def set_positions(self, number_of_stations=1, station_separation=2000,
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
        for base_pos in station_positions:
            for str_index in range(strings_per_station):
                angle = str_index/strings_per_station * 2*np.pi
                x = base_pos[0] + string_separation*np.cos(angle)
                y = base_pos[1] + string_separation*np.sin(angle)
                for ant_index in range(antennas_per_string):
                    z = lowest_antenna + ant_index*antenna_separation
                    self.antenna_positions.append((x,y,z))

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
                AntennaClass(name=AntennaClass.__name__, position=pos,
                             power_threshold=power_threshold,
                             amplification=amplification,
                             noisy=noisy)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))



class ARATriangleDetector(Detector):
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
    def set_positions(self, number_of_stations=1, station_separation=2000,
                      outrigger_strings_per_station=3, station_diameter=40,
                      outrigger_antenna_pairs=4, outrigger_pair_separation=30,
                      outrigger_inter_pair_separation=1,
                      outrigger_highest_antenna=10,
                      central_vpol_antennas=8, central_highest_vpol=60, central_vpol_separation=1, central_hpol_antennas=8,
                      central_highest_hpol=40, central_hpol_separation=1):
        self.antenna_positions = []
        self.antenna_types = []

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
        for base_pos in station_positions:
            # Set up outrigger strings
            for str_index in range(outrigger_strings_per_station):
                angle = str_index/outrigger_strings_per_station * 2*np.pi
                x = base_pos[0] + station_diameter/2*np.cos(angle)
                y = base_pos[1] + station_diameter/2*np.sin(angle)
                for pair_index in range(outrigger_antenna_pairs):
                    z = (outrigger_highest_antenna -
                         pair_index*outrigger_pair_separation)
                    self.antenna_positions.append((x,y,z))
                    self.antenna_types.append("H")
                    z -= outrigger_inter_pair_separation
                    self.antenna_positions.append((x,y,z))
                    self.antenna_types.append("V")
            # Set up central phased-array string
            x = base_pos[0]
            y = base_pos[1]
            for ant_index in range(central_hpol_antennas):
                z = central_highest_hpol - ant_index*central_hpol_separation
                self.antenna_positions.append((x,y,z))
                self.antenna_types.append("H")
            for ant_index in range(central_vpol_antennas):
                z = central_highest_vpol - ant_index*central_vpol_separation
                self.antenna_positions.append((x,y,z))
                self.antenna_types.append("V")

    def build_antennas(self, power_threshold, amplification=1,
                       naming_scheme=lambda i, ant: ant.name[:4]+"_"+str(i),
                       noisy=True):
        """Sets up ARAAntennas at the positions stored in the class.
        Takes as arguments the power threshold, amplification, and whether to
        add noise to the waveforms.
        Other optional arguments include a naming scheme and orientation scheme
        which are functions taking the antenna index i and the antenna object.
        The naming scheme should return the name and the orientation scheme
        should return the orientation z-axis and x-axis of the antenna."""
        self.antennas = []
        for i, (pos, pol) in enumerate(zip(self.antenna_positions,
                                           self.antenna_types)):
            if pol=="H":
                AntennaClass = HpolAntenna
            elif pol=="V":
                AntennaClass = VpolAntenna
            else:
                raise ValueError("Unknown antenna type {}".format(pol))
            self.antennas.append(
                AntennaClass(name=AntennaClass.__name__, position=pos,
                             power_threshold=power_threshold,
                             amplification=amplification,
                             noisy=noisy)
            )
        for i, ant in enumerate(self.antennas):
            ant.name = str(naming_scheme(i, ant))
