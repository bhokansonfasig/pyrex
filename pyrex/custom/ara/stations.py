"""
Module containing classes representing deployed ARA stations.

Designed to match the coordinate specifications of the existing ARA stations
deployed at South Pole.

"""

import logging
import os
import os.path
import sqlite3
import numpy as np
from pyrex.internal_functions import normalize
from pyrex.antenna import Antenna
from pyrex.detector import Detector
from .antenna import ARA_DATA_DIR, HpolAntenna, VpolAntenna
from .detector import ARAString, RegularStation

logger = logging.getLogger(__name__)


def _read_antenna_database(station, database):
    """
    Get antenna information for the given ARA station from the database.

    Parameters
    ----------
    station : str
        Name of the station's table in the database.
    database : str
        Name of the sqlite database containing antenna information.

    Returns
    -------
    names : list of str
        List of antenna names (consisting of their borehole and antenna type).
    positions : list of tuple
        List of antenna positions (m) in the local station coordinates.

    """
    names = []
    positions = []
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    for row in c.execute("SELECT * FROM "+str(station)):
        names.append(row['holeName']+' '+row['antName'])
        positions.append((row['antLocationX'],
                          row['antLocationY'],
                          row['antLocationZ']))
    conn.close()
    return names, positions


def _read_station_database(station, database):
    """
    Get station information for the given ARA station from the database.

    Parameters
    ----------
    station : str
        Name of the station as recorded in the database.
    database : str
        Name of the sqlite database containing station information.

    Returns
    -------
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).
    local_coords : ndarray
        Matrix of local coordinate system axes in global coordinates.

    Raises
    ------
    ValueError
        If the station name doesn't appear in the database.

    """
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    for row in c.execute("SELECT * FROM ARA"):
        if row['stationName']==station:
            conn.close()
            coords = np.array([
                [row['local00'], row['local01'], row['local02']],
                [row['local10'], row['local11'], row['local12']],
                [row['local20'], row['local21'], row['local22']],
            ])
            return (row['stationEasting'],
                    row['stationNorthing'],
                    row['stationElevation'],
                    coords)
    raise ValueError("Couldn't find station '"+str(station)+"' in database")


ARA_DATABASE_PATH = os.environ.get("ARA_UTIL_INSTALL_DIR")
if ARA_DATABASE_PATH is None:
    ARA_DATABASE_PATH = ARA_DATA_DIR
else:
    ARA_DATABASE_PATH = os.path.join(ARA_DATABASE_PATH, 'share', 'araCalib')
logger.info("Getting ARA station coordinates from %s", ARA_DATABASE_PATH)

ARA_ANTENNA_DB = os.path.join(ARA_DATABASE_PATH, 'AntennaInfo.sqlite')
ARA_STATION_DB = os.path.join(ARA_DATABASE_PATH, 'AraArrayCoords.sqlite')
ARA_CALPULSER_DB = os.path.join(ARA_DATABASE_PATH, 'CalPulserInfo.sqlite')


def _convert_local_to_global(position, local_coords):
    """
    Convert local "station" coordinates into global "array" coordinates.

    Parameters
    ----------
    position : array_like
        Cartesian position in local "station" coordinates to be transformed.
    local_coords : array_like
        Matrix of local coordinate system axes in global coordinates.

    Returns
    -------
    global_position : ndarray
        Cartesian position transformed to the global "array" coordinates.

    """
    local_x = normalize(local_coords[0])
    global_x = (1, 0, 0)
    # Find angle between x-axes and rotation axis perpendicular to both x-axes
    angle = np.arccos(np.dot(local_x, global_x))
    axis = normalize(np.cross(global_x, local_x))
    # Form rotation matrix
    cos = np.cos(angle)
    sin = np.sin(angle)
    ux, uy, uz = axis
    rot = np.array([
        [cos + ux**2*(1-cos), ux*uy*(1-cos) - uz*sin, ux*uz*(1-cos) + uy*sin],
        [uy*ux*(1-cos) + uz*sin, cos + uy**2*(1-cos), uy*uz*(1-cos) - ux*sin],
        [uz*ux*(1-cos) - uy*sin, uz*uy*(1-cos) + ux*sin, cos + uz**2*(1-cos)],
    ])
    # Rotate position to new axes
    return np.dot(rot, position)


def _get_deployed_strings(station, x=0, y=0, z=0, local_coords=None):
    """
    Create DeployedString instances for the given station.

    The positions of antennas can be returned in either the local "station"
    coordinates or the global "array" coordinates.

    Parameters
    ----------
    station : str
        Name of the station as recorded in the antenna information database.
    x : float, optional
        Cartesian x-position (m) of the station center.
    y : float, optional
        Cartesian y-position (m) of the station center.
    z : float, optional
        Cartesian z-position offset (m) from the nominal surface.
    local_coords : array_like or None, optional
        Matrix of local coordinate system axes in global coordinates. If
        `None`, the antenna positions will be set in the local coordinates.
        Otherwise, this matrix will be used to transform the antenna positions
        to global coordinates (in which case specifying `x`, `y`, and `z` is
        crucial).

    Returns
    -------
    list of DeployedString
        List of strings which make up the station.

    """
    subsets = []
    data_names, data_positions = _read_antenna_database(station,
                                                        ARA_ANTENNA_DB)
    strings = sorted(list(set([name.split()[0] for name in data_names
                               if name.startswith("BH")])))
    positions = {string: [] for string in strings}
    names = {string: [] for string in strings}
    types = {string: [] for string in strings}
    transform_coordinates = local_coords is not None
    for name, pos in zip(data_names, data_positions):
        for string in strings:
            if name.startswith(string):
                if transform_coordinates:
                    pos = _convert_local_to_global(pos, local_coords)
                positions[string].append((pos[0]+x, pos[1]+y, pos[2]+z))
                names[string].append(name)
                if "VPol" in name:
                    types[string].append(VpolAntenna)
                elif "HPol" in name:
                    types[string].append(HpolAntenna)
                else:
                    raise ValueError("No known antenna type for '"+
                                     str(name)+"'")
    for string in strings:
        subsets.append(
            DeployedString(positions[string], types[string], names[string])
        )
    return subsets


def _get_calpulser_strings(station, x=0, y=0, z=0, local_coords=None):
    """
    Create CalpulserString instances for the given station.

    The positions of antennas can be returned in either the local "station"
    coordinates or the global "array" coordinates.

    Parameters
    ----------
    station : str
        Name of the station as recorded in the antenna information database.
    x : float, optional
        Cartesian x-position (m) of the station center.
    y : float, optional
        Cartesian y-position (m) of the station center.
    z : float, optional
        Cartesian z-position offset (m) from the nominal surface.
    local_coords : array_like or None, optional
        Matrix of local coordinate system axes in global coordinates. If
        `None`, the antenna positions will be set in the local coordinates.
        Otherwise, this matrix will be used to transform the antenna positions
        to global coordinates (in which case specifying `x`, `y`, and `z` is
        crucial).

    Returns
    -------
    list of CalpulserString
        List of strings which make up the station.

    """
    subsets = []
    data_names, data_positions = _read_antenna_database(station,
                                                        ARA_CALPULSER_DB)
    strings = sorted(list(set([name.split()[0] for name in data_names
                               if name.startswith("BH")])))
    positions = {string: [] for string in strings}
    names = {string: [] for string in strings}
    transform_coordinates = local_coords is not None
    for name, pos in zip(data_names, data_positions):
        for string in strings:
            if name.startswith(string):
                if transform_coordinates:
                    pos = _convert_local_to_global(pos, local_coords)
                positions[string].append((pos[0]+x, pos[1]+y, pos[2]+z))
                names[string].append(name)
    for string in strings:
        subsets.append(
            CalpulserString(positions[string], names[string])
        )
    return subsets


class DeployedString(ARAString):
    """
    String of ARA Hpol and Vpol antennas as deployed at South Pole.

    Sets the positions of antennas on string based on the parameters. Once the
    antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    positions : array_like
        Array of Cartesian positions (m) for each antenna on the string.
    types : array_like
        List of classes to be used for construction of each antenna.
    names : array_like
        List of names to be given to the antennas.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.

    Notes
    -----
    This class is designed to be the lowest subset level of a detector. It can
    (and should) be used for the subsets of some other ``Detector`` subclass
    to build up a full detector. Then when its "parent" is iterated, the
    instances of this class will be iterated as though they were all part of
    one flat list.

    """
    def set_positions(self, positions, types, names):
        """
        Generates antenna positions along the string.

        Parameters
        ----------
        positions : array_like
            Array of Cartesian positions (m) for each antenna on the string.
        types : array_like
            List of classes to be used for construction of each antenna.
        names : array_like
            List of names to be given to the antennas.

        """
        self._antenna_types = []
        self._antenna_names = []
        if len(positions)!=len(types) or len(positions)!=len(names):
            raise ValueError("Lengths of input arrays must match")
        for ant_pos, ant_type, ant_name in zip(positions, types, names):
            self.antenna_positions.append(tuple(ant_pos))
            self._antenna_types.append(ant_type)
            self._antenna_names.append(ant_name)

    def build_antennas(self, *args, **kwargs):
        """
        Creates antenna objects at the set antenna positions.

        Antenna types built and their names are based on the values given on
        class initialization.

        Parameters
        ----------
        power_threshold : float
            Power threshold for antenna trigger conditions.
        amplification : float, optional
            Amplification to be applied to antenna signals.
        noisy : boolean, optional
            Whether or not the antenna should add noise to incoming signals.
        unique_noise_waveforms : int, optional
            The number of expected noise waveforms needed for each received
            signal to have its own noise (per antenna).

        """
        super().build_antennas(
            *args,
            naming_scheme=lambda i, ant: self._antenna_names[i],
            class_scheme=lambda i: self._antenna_types[i],
            **kwargs
        )



class CalpulserString(Detector):
    """
    String of ARA calibration pulser antennas.

    Sets the positions of antennas on string based on the parameters. Once the
    antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    positions : array_like
        Array of Cartesian positions (m) for each antenna on the string.
    names : array_like
        List of names to be given to the antennas.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    Notes
    -----
    This class is designed to be the lowest subset level of a detector. It can
    (and should) be used for the subsets of some other ``Detector`` subclass
    to build up a full detector. Then when its "parent" is iterated, the
    instances of this class will be iterated as though they were all part of
    one flat list.

    """
    def set_positions(self, positions, names):
        """
        Generates antenna positions along the string.

        Parameters
        ----------
        positions : array_like
            Array of Cartesian positions (m) for each antenna on the string.
        names : array_like
            List of names to be given to the antennas.

        """
        self._antenna_names = []
        if len(positions)!=len(names):
            raise ValueError("Lengths of input arrays must match")
        for ant_pos, ant_name in zip(positions, names):
            self.antenna_positions.append(tuple(ant_pos))
            self._antenna_names.append(ant_name)

    def build_antennas(self, *args, **kwargs):
        """
        Creates antenna objects at the set antenna positions.

        Antenna names are based on the values given on class initialization.

        Parameters
        ----------
        antenna_class : optional
            Class to be used for the antennas.

        """
        if 'antenna_class' not in kwargs:
            kwargs['antenna_class'] = Antenna
        super().build_antennas(*args, **kwargs)
        for i, ant in enumerate(self.subsets):
            ant.name = self._antenna_names[i]


class Calpulsers(Detector):
    """
    Group of ARA calibration pulser antenna strings.

    Sets the positions of antennas on string based on the parameters. Once the
    antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    positions : array_like
        Array of Cartesian positions (m) for each antenna on the string.
    names : array_like
        List of names to be given to the antennas.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    def set_positions(self, station, x=0, y=0, z=0, local_coords=None):
        """
        Generates antenna positions along the string.

        Parameters
        ----------
        station : str
            Name of the station as recorded in the antenna information database.
        x : float, optional
            Cartesian x-position (m) of the station center.
        y : float, optional
            Cartesian y-position (m) of the station center.
        z : float, optional
            Cartesian z-position offset (m) from the nominal surface.
        local_coords : array_like or None, optional
            Matrix of local coordinate system axes in global coordinates. If
            `None`, the antenna positions will be set in the local coordinates.
            Otherwise, this matrix will be used to transform the antenna
            positions to global coordinates (in which case specifying `x`, `y`,
            and `z` is crucial).

        """
        self.subsets.extend(
            _get_calpulser_strings(station, x, y, z, local_coords)
        )



class ARA01(RegularStation):
    """
    Station geometry representing the deployed station ARA01.

    Sets the positions of strings around the station based on the parameters.
    Once the antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    global_coords : bool, optional
        Whether the station should be positioned in global coordinates. If
        `True`, the station will be set in the global "array" coordinate system
        with appropriate relative positions to other ARA stations. In this case
        the x-positions of antennas are their Easting coordinates and the
        y-positions are their Northing coordinates. If `False`, the station
        will be set in the local "station" coordinate system with the station
        centered around zero in x and y.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.
    pyrex.custom.ara.ARAString : String of ARA Hpol and Vpol antennas.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    easting, northing, elevation, _local_coords = _read_station_database(
        'ARA1', ARA_STATION_DB
    )

    def set_positions(self, global_coords=False):
        """
        Generates antenna positions around the station.

        Parameters
        ----------
        global_coords : bool, optional
            Whether the station should be positioned in global coordinates. If
            `True`, the station will be set in the global "array" coordinate
            system with appropriate relative positions to other ARA stations.
            In this case the x-positions of antennas are their Easting
            coordinates and the y-positions are their Northing coordinates. If
            `False`, the station will be set in the local "station" coordinate
            system with the station centered around zero in x and y.

        """
        if global_coords:
            self.subsets.extend(
                _get_deployed_strings(station='ARA01',
                                      x=self.easting,
                                      y=self.northing,
                                      z=self.elevation,
                                      local_coords=self._local_coords)
            )
            self.calpulsers = Calpulsers(station='ARA01',
                                         x=self.easting,
                                         y=self.northing,
                                         z=self.elevation,
                                         local_coords=self._local_coords)
            self.calpulsers.build_antennas()
        else:
            self.subsets.extend(
                _get_deployed_strings(station='ARA01', x=0, y=0, z=0,
                                      local_coords=None)
            )
            self.calpulsers = Calpulsers(station='ARA01', x=0, y=0, z=0,
                                         local_coords=None)
            self.calpulsers.build_antennas()


class ARA02(RegularStation):
    """
    Station geometry representing the deployed station ARA01.

    Sets the positions of strings around the station based on the parameters.
    Once the antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    global_coords : bool, optional
        Whether the station should be positioned in global coordinates. If
        `True`, the station will be set in the global "array" coordinate system
        with appropriate relative positions to other ARA stations. In this case
        the x-positions of antennas are their Easting coordinates and the
        y-positions are their Northing coordinates. If `False`, the station
        will be set in the local "station" coordinate system with the station
        centered around zero in x and y.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.
    pyrex.custom.ara.ARAString : String of ARA Hpol and Vpol antennas.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    easting, northing, elevation, _local_coords = _read_station_database(
        'ARA2', ARA_STATION_DB
    )

    def set_positions(self, global_coords=False):
        """
        Generates antenna positions around the station.

        Parameters
        ----------
        global_coords : bool, optional
            Whether the station should be positioned in global coordinates. If
            `True`, the station will be set in the global "array" coordinate
            system with appropriate relative positions to other ARA stations.
            In this case the x-positions of antennas are their Easting
            coordinates and the y-positions are their Northing coordinates. If
            `False`, the station will be set in the local "station" coordinate
            system with the station centered around zero in x and y.

        """
        if global_coords:
            self.subsets.extend(
                _get_deployed_strings(station='ARA02',
                                      x=self.easting,
                                      y=self.northing,
                                      z=self.elevation,
                                      local_coords=self._local_coords)
            )
            self.calpulsers = Calpulsers(station='ARA02',
                                         x=self.easting,
                                         y=self.northing,
                                         z=self.elevation,
                                         local_coords=self._local_coords)
            self.calpulsers.build_antennas()
        else:
            self.subsets.extend(
                _get_deployed_strings(station='ARA02', x=0, y=0, z=0,
                                      local_coords=None)
            )
            self.calpulsers = Calpulsers(station='ARA02', x=0, y=0, z=0,
                                         local_coords=None)
            self.calpulsers.build_antennas()


class ARA03(RegularStation):
    """
    Station geometry representing the deployed station ARA03.

    Sets the positions of strings around the station based on the parameters.
    Once the antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    global_coords : bool, optional
        Whether the station should be positioned in global coordinates. If
        `True`, the station will be set in the global "array" coordinate system
        with appropriate relative positions to other ARA stations. In this case
        the x-positions of antennas are their Easting coordinates and the
        y-positions are their Northing coordinates. If `False`, the station
        will be set in the local "station" coordinate system with the station
        centered around zero in x and y.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.
    pyrex.custom.ara.ARAString : String of ARA Hpol and Vpol antennas.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    easting, northing, elevation, _local_coords = _read_station_database(
        'ARA3', ARA_STATION_DB
    )

    def set_positions(self, global_coords=False):
        """
        Generates antenna positions around the station.

        Parameters
        ----------
        global_coords : bool, optional
            Whether the station should be positioned in global coordinates. If
            `True`, the station will be set in the global "array" coordinate
            system with appropriate relative positions to other ARA stations.
            In this case the x-positions of antennas are their Easting
            coordinates and the y-positions are their Northing coordinates. If
            `False`, the station will be set in the local "station" coordinate
            system with the station centered around zero in x and y.

        """
        if global_coords:
            self.subsets.extend(
                _get_deployed_strings(station='ARA03',
                                      x=self.easting,
                                      y=self.northing,
                                      z=self.elevation,
                                      local_coords=self._local_coords)
            )
            self.calpulsers = Calpulsers(station='ARA03',
                                         x=self.easting,
                                         y=self.northing,
                                         z=self.elevation,
                                         local_coords=self._local_coords)
            self.calpulsers.build_antennas()
        else:
            self.subsets.extend(
                _get_deployed_strings(station='ARA03', x=0, y=0, z=0,
                                      local_coords=None)
            )
            self.calpulsers = Calpulsers(station='ARA03', x=0, y=0, z=0,
                                         local_coords=None)
            self.calpulsers.build_antennas()


class ARA04(RegularStation):
    """
    Station geometry representing the deployed station ARA04.

    Sets the positions of strings around the station based on the parameters.
    Once the antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    global_coords : bool, optional
        Whether the station should be positioned in global coordinates. If
        `True`, the station will be set in the global "array" coordinate system
        with appropriate relative positions to other ARA stations. In this case
        the x-positions of antennas are their Easting coordinates and the
        y-positions are their Northing coordinates. If `False`, the station
        will be set in the local "station" coordinate system with the station
        centered around zero in x and y.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.
    pyrex.custom.ara.ARAString : String of ARA Hpol and Vpol antennas.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    easting, northing, elevation, _local_coords = _read_station_database(
        'ARA4', ARA_STATION_DB
    )

    def set_positions(self, global_coords=False):
        """
        Generates antenna positions around the station.

        Parameters
        ----------
        global_coords : bool, optional
            Whether the station should be positioned in global coordinates. If
            `True`, the station will be set in the global "array" coordinate
            system with appropriate relative positions to other ARA stations.
            In this case the x-positions of antennas are their Easting
            coordinates and the y-positions are their Northing coordinates. If
            `False`, the station will be set in the local "station" coordinate
            system with the station centered around zero in x and y.

        """
        if global_coords:
            self.subsets.extend(
                _get_deployed_strings(station='ARA04',
                                      x=self.easting,
                                      y=self.northing,
                                      z=self.elevation,
                                      local_coords=self._local_coords)
            )
            self.calpulsers = Calpulsers(station='ARA04',
                                         x=self.easting,
                                         y=self.northing,
                                         z=self.elevation,
                                         local_coords=self._local_coords)
            self.calpulsers.build_antennas()
        else:
            self.subsets.extend(
                _get_deployed_strings(station='ARA04', x=0, y=0, z=0,
                                      local_coords=None)
            )
            self.calpulsers = Calpulsers(station='ARA04', x=0, y=0, z=0,
                                         local_coords=None)
            self.calpulsers.build_antennas()


class ARA05(RegularStation):
    """
    Station geometry representing the deployed station ARA05.

    Sets the positions of strings around the station based on the parameters.
    Once the antennas have been built with `build_antennas`, the object can be
    directly iterated over to iterate over the antennas (as if the object were
    just a list of the antennas).

    Parameters
    ----------
    global_coords : bool, optional
        Whether the station should be positioned in global coordinates. If
        `True`, the station will be set in the global "array" coordinate system
        with appropriate relative positions to other ARA stations. In this case
        the x-positions of antennas are their Easting coordinates and the
        y-positions are their Northing coordinates. If `False`, the station
        will be set in the local "station" coordinate system with the station
        centered around zero in x and y.

    Attributes
    ----------
    antenna_positions : list
        List (potentially with sub-lists) of the positions of the antennas
        generated by the `set_positions` method.
    subsets : list
        List of the antenna or detector objects which make up the detector.
    test_antenna_positions : boolean
        Class attribute for whether or not an error should be raised if antenna
        positions are found above the surface of the ice (where simulation
        behavior is ill-defined). Defaults to ``True``.
    easting : float
        Station nominal Easting coordinate (m).
    northing : float
        Station nominal Northing coordinate (m).
    elevation : float
        Station nominal elevation (m).

    Raises
    ------
    ValueError
        If ``test_antenna_positions`` is ``True`` and an antenna is found to be
        above the ice surface.

    See Also
    --------
    pyrex.custom.ara.HpolAntenna : ARA Hpol (“quad-slot”) antenna system with
                                   front-end processing.
    pyrex.custom.ara.VpolAntenna : ARA Vpol (“bicone” or “birdcage”) antenna
                                   system with front-end processing.
    pyrex.custom.ara.ARAString : String of ARA Hpol and Vpol antennas.

    Notes
    -----
    This class is designed to have string-like objects (which are subclasses of
    ``Detector``) as its `subsets`. Then whenever an object of this class is
    iterated, all the antennas of its strings will be yielded as in a 1D list.

    """
    easting, northing, elevation, _local_coords = _read_station_database(
        'ARA5', ARA_STATION_DB
    )

    def set_positions(self, global_coords=False):
        """
        Generates antenna positions around the station.

        Parameters
        ----------
        global_coords : bool, optional
            Whether the station should be positioned in global coordinates. If
            `True`, the station will be set in the global "array" coordinate
            system with appropriate relative positions to other ARA stations.
            In this case the x-positions of antennas are their Easting
            coordinates and the y-positions are their Northing coordinates. If
            `False`, the station will be set in the local "station" coordinate
            system with the station centered around zero in x and y.

        """
        if global_coords:
            self.subsets.extend(
                _get_deployed_strings(station='ARA05',
                                      x=self.easting,
                                      y=self.northing,
                                      z=self.elevation,
                                      local_coords=self._local_coords)
            )
            self.calpulsers = Calpulsers(station='ARA05',
                                         x=self.easting,
                                         y=self.northing,
                                         z=self.elevation,
                                         local_coords=self._local_coords)
            self.calpulsers.build_antennas()
        else:
            self.subsets.extend(
                _get_deployed_strings(station='ARA05', x=0, y=0, z=0,
                                      local_coords=None)
            )
            self.calpulsers = Calpulsers(station='ARA05', x=0, y=0, z=0,
                                         local_coords=None)
            self.calpulsers.build_antennas()
