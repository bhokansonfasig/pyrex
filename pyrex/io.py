"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

import datetime
import inspect
import logging
from pkg_resources import parse_version
import h5py
import numpy as np
from pyrex.__about__ import __version__
from pyrex.internal_functions import get_from_enum
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.particle import Particle, Event

logger = logging.getLogger(__name__)


H5PY_V2_STRINGS = parse_version(h5py.__version__) < parse_version('3.0.0')


class BaseReader:
    """
    Base class for reading data from a file.

    Works as a context manager and allows for reading simulation/real data or
    analysis-level data from the given file.

    Parameters
    ----------
    filename : str
        File name to open in read mode.

    Attributes
    ----------
    is_open

    """
    def __init__(self, filename):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        """
        Open the file for reading.

        """
        raise NotImplementedError

    def close(self):
        """
        Close the file.

        """
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    @property
    def is_open(self):
        """
        Boolean of whether the file is currently open.

        """
        raise NotImplementedError


class BaseWriter:
    """
    Base class for writing data to a file.

    Works as a context manager and allows for writing simulation/real data or
    analysis-level data to the given file.

    Parameters
    ----------
    filename : str
        File name to open in write mode. This will overwrite an existing file
        of the same name.

    Attributes
    ----------
    is_open
    has_detector

    """
    def __init__(self, filename, **kwargs):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        """
        Open the file for writing.

        """
        raise NotImplementedError

    def close(self):
        """
        Close the file.

        """
        raise NotImplementedError

    @property
    def is_open(self):
        """
        Boolean of whether the file is currently open.

        """
        raise NotImplementedError

    def set_detector(self, detector):
        """
        Link the given `detector` to the file.

        This `detector` is used for writing antenna data and metadata, as well
        as for reading in waveforms and other data. Typically will be the
        same detector object used in the `EventKernel`.

        Parameters
        ----------
        detector
            Detector object or list of antennas for this output file.

        """
        raise NotImplementedError

    @property
    def has_detector(self):
        """
        Boolean of whether the file has a linked detector.

        """
        raise NotImplementedError



class HDF5Base:
    """
    Base class for readers and writers of hdf5 files.

    Defines parameters and methods to be used by the hdf5 I/O classes based on
    the version number of the file that is being worked with.

    Parameters
    ----------
    file_version_major : int
        Major portion of the file version number (X in X.Y).
    file_version_minor : int
        Minor portion of the file version number (Y in X.Y).

    Attributes
    ----------
    _file_version_major : int
        Major portion of the file version number (X in X.Y).
    _file_version_minor : int
        Minor portion of the file version number (Y in X.Y).

    """
    def __init__(self, file_version_major, file_version_minor):
        self._file_version_major = file_version_major
        self._file_version_minor = file_version_minor

    def _decode_attr(self, attribute):
        """
        Decode attributes from the file to string values.

        Parameters
        ----------
        attribute : str or bytes
            Raw value of an attribute from a `DATASET.attrs` list.

        Returns
        -------
        str
            Attribute value decoded into the `str` type.

        Notes
        -----
        If the file was written using h5py versions below 3.0.0 (file version
        1.0), this function decodes the byte-typed attribute. If written using
        h5py versions above 3.0.0 (file version 1.1+), this function simply
        passes along the string-typed attribute.

        """
        if self._file_version_major==1 and self._file_version_minor==0:
            return bytes.decode(attribute)
        else:
            return attribute

    def _encode_attr(self, attribute):
        """
        Encode attribute string values for writing to the file.

        Parameters
        ----------
        attribute : str
            String value to be stored as an attribute in a `DATASET.attrs` list.

        Returns
        -------
        str or bytes
            Attribute value encoded into the appropriate type for storage.

        Notes
        -----
        If the file is being written using h5py versions below 3.0.0 (file
        version 1.0), this function encodes to a byte-typed attribute. If it is
        being written using h5py versions above 3.0.0 (file version 1.1+), this
        function simply passes along the string-typed attribute.

        """
        if self._file_version_major==1 and self._file_version_minor==0:
            return str.encode(attribute)
        else:
            return attribute

    def _decode_string_data(self, data):
        """
        Convert data from a dataset of string values to `str`.

        Parameters
        ----------
        data : [array of] bytes or str
            Data pulled directly from the dataset.

        Returns
        -------
        str
            Data in the `str` type.

        Notes
        -----
        If the file was written using h5py versions below 3.0.0 (file version
        1.0), this function simply passes along the data since it is already
        a `str` value. If written using h5py versions above 3.0.0 (file
        version 1.1+), this function decodes the byte-typed string to `str`.

        """
        if self._file_version_major==1 and self._file_version_minor==0:
            return data
        else:
            if isinstance(data, np.ndarray):
                return np.array([bytes.decode(b) for b in data.flat]).reshape(data.shape)
            else:
                return bytes.decode(data)

    def _dataset_locations(self):
        """
        Get the file locations of datasets and metadata groups.

        Since dataset locations may change with different file versions, this
        method provides a common interface.

        Returns
        -------
        dict
            Dictionary of file locations as values with common corresponding
            keys.

        """
        # May wish to adjust behavior based on file version
        # (self._file_version_major and self._file_version_minor)

        # If the location is a group which has float and string datasets,
        # the key should end with "_meta" (for compatibility with the reader)
        locations = {}
        locations['file_meta'] = "/file_metadata"
        locations['indices'] = "/event_indices"
        locations['waveforms'] = "/data/waveforms"
        locations['triggers'] = "/data/triggers"
        locations['antennas'] = "/data/antennas"
        locations['particles_meta'] = "/monte_carlo_data/particles"
        locations['antennas_meta'] = "/monte_carlo_data/antennas"
        locations['rays_meta'] = "/monte_carlo_data/rays"
        locations['mc_triggers'] = "/monte_carlo_data/triggers"
        locations['noise'] = "/monte_carlo_data/noise"
        return locations

    @staticmethod
    def _generate_location_names(file, existing_locations):
        """
        Generate unique keys based on dataset locations in a file.

        Parameters
        ----------
        file
            hdf5 file object to be read.
        existing_locations : dict
            Existing dictionary of location names with values corresponding to
            their actual location in the file. These names will be avoided when
            creating keys for the other datasets in the file.

        Returns
        -------
        dict
            Dictionary of file locations as values with corresponding keys.

        """
        locations = {}
        locations.update(existing_locations)
        unregistered = []
        def get_unregistered_locations(name):
            if not name.startswith('/'):
                name = '/'+name
            final_name = name.split('/')[-1]
            is_dataset = (isinstance(file[name], h5py.Dataset)
                          and final_name!='str' and final_name!='float')
            is_metagroup = (name+'/str' in file and
                             isinstance(file[name+'/str'], h5py.Dataset) and
                             name+'/float' in file and
                             isinstance(file[name+'/float'], h5py.Dataset))
            if (is_dataset or is_metagroup) and name not in locations.values():
                unregistered.append(name)
        file.visit(get_unregistered_locations)

        def get_level_key(name, level):
            return "_".join(name.split('/')[-level-1:])

        def get_unique_key(name):
            for level in range(name.count('/')):
                key = get_level_key(name, level)
                unique = True
                if key in locations.keys():
                    continue
                unique = True
                for other in unregistered:
                    if other!=name and get_level_key(other, level)==key:
                        unique = False
                        break
                if unique:
                    return key
            raise ValueError("Couldn't generate a unique key for "+str(name))

        for value in unregistered:
            try:
                key = get_unique_key(value)
            except ValueError:
                logger.info("Unable to set unique location name for %s", value)
            else:
                locations[key] = value

        return locations

    def _analysis_location(self, name):
        """
        Transform the given name into a valid analysis location.

        Since dataset locations may change with different file versions, this
        method provides a common interface.

        Parameters
        ----------
        name : str
            Name of the dataset or group. May contain the full path to the
            dataset or group location.

        Returns
        -------
        str
            Dataset/group name moved to a valid analysis location within the
            file if necessary.

        """
        # May wish to adjust behavior based on file version
        # (self._file_version_major and self._file_version_minor)
        if not name.startswith("/"):
            name = "/"+name
        parts = name.split("/")
        if parts[1] not in ["monte_carlo_data", "data"]:
            parts = [parts[0]] + ["monte_carlo_data"] + parts[1:]
        if parts[2]!="analysis":
            parts = parts[:2] + ["analysis"] + parts[2:]
        new_name = "/".join(parts)
        if new_name!=name:
            logger.info("%s not a valid analysis path, adjusting to %s",
                        name, new_name)
        return new_name

    def _read_metadata_to_dicts(self, data, name, index=None,
                                str_keys=None, float_keys=None):
        """
        Read data from a metadata group into a dictionary for convenience.

        Pulls data from the 'str' and 'float' datasets of the metadata group
        into a common dictionary (or array of dictionaries) so the value types
        don't need to be considered.

        Parameters
        ----------
        data
            hdf5 file object or data dict containing the data.
        name : str
            Location and name of the metadata group. Must contain the full path
            to the location and name of the metadata group if reading from a
            file. Must be the key prefix (without '_float' or '_str') if
            reading from a dictionary.
        index : int or slice, optional
            Index or slice to be applied on the first dimension of the 'str'
            and 'float' datasets. No slicing of the datasets by default.
        str_keys : list of str, optional
            List of keys corresponding to columns in the 'str' dataset.
        float_keys : list of str, optional
            List of keys corresponding to columns in the 'float' dataset.

        Returns
        -------
        dict or array_like of dict
            Dictionaries containing the information in the metadata datasets
            with column names as keys.

        """
        if isinstance(data, dict):
            str_metadata = data[name+'_str']
            float_metadata = data[name+'_float']
        else:
            str_metadata = data[name]['str']
            float_metadata = data[name]['float']

        if index is None:
            str_table = str_metadata
            float_table = float_metadata
        else:
            str_table = str_metadata[index]
            float_table = float_metadata[index]

        if str_keys is None:
            str_keys = [self._decode_attr(key) for key
                        in str_metadata.attrs['keys']]
        if float_keys is None:
            float_keys = [self._decode_attr(key) for key
                          in float_metadata.attrs['keys']]

        if str_table.ndim!=float_table.ndim:
            raise ValueError("Metadata group '"+name+"' not readable")

        ndim = str_table.ndim
        key_dim = -1
        if ndim==0:
            return {}

        if (str_table.shape[key_dim]!=len(str_keys) or
                float_table.shape[key_dim]!=len(float_keys)):
            raise ValueError("Metadata group '"+name+"' not readable")

        # Recursively pull out list (of lists ...) of dictionaries
        def get_dicts_recursive(dimension, str_data, float_data):
            if dimension==key_dim%ndim:
                meta_dict = {}
                for j, key in enumerate(str_keys):
                    meta_dict[key] = self._decode_string_data(str_data[j])
                for j, key in enumerate(float_keys):
                    meta_dict[key] = float_data[j]
                return meta_dict
            else:
                if str_data.shape[0]!=float_data.shape[0]:
                    raise ValueError("Metadata group '"+name+"' not readable")
                return [
                    get_dicts_recursive(dimension+1, str_data[i], float_data[i])
                    for i in range(str_data.shape[0])
                ]

        return get_dicts_recursive(0, str_table, float_table)

    def _read_dataset_to_dicts(self, data, name, index=None, keys=None):
        """
        Read data from a dataset into a dictionary for convenience.

        Pulls data from the dataset into a dictionary (or array of
        dictionaries) with column names as keys.

        Parameters
        ----------
        data
            hdf5 file object or data dict containing the data.
        name : str
            Location and name of the dataset. Must contain the full path to the
            location and name of the dataset if reading from a file. Must be
            the key if reading from a dictionary.
        index : int or slice, optional
            Index or slice to be applied on the first dimension of the dataset.
            No slicing of the dataset by default.
        keys : list of str, optional
            List of keys corresponding to columns in the dataset.

        Returns
        -------
        dict or array_like of dict
            Dictionaries containing the information in the dataset with column
            names as keys.

        """
        if isinstance(data, dict):
            dataset = data[name]
        else:
            dataset = data[name]

        if index is None:
            table = dataset
        else:
            table = dataset[index]

        if keys is None:
            keys = [self._decode_attr(key) for key in dataset.attrs['keys']]

        ndim = table.ndim
        key_dim = -1

        # Recursively pull out list (of lists ...) of dictionaries
        def get_dicts_recursive(dimension, table_data):
            if dimension == key_dim % ndim:
                meta_dict = {}
                for j, key in enumerate(keys):
                    meta_dict[key] = table_data[j]
                return meta_dict
            else:
                return [
                    get_dicts_recursive(
                        dimension+1, table[i])
                    for i in range(table.shape[0])
                ]

        return get_dicts_recursive(0, table)

    @staticmethod
    def _get_bool_dict(file, locations):
        """
        Get a boolean dictionary of the existence of locations in the file.

        For each location in locations, the value of the dictionary will be
        whether the location exists and has a size larger than zero.

        Parameters
        ----------
        file
            hdf5 file object to be read.
        locations : dict
            Dictionary of location names with values corresponding to their
            actual location in the file.

        Returns
        -------
        dict
            Dictionary of boolean values denoting the existence of locations.

        """
        return {key: group in file and file[group].size>0
                for key, group in locations.items()}

    def _get_keys_dict(self, file, group):
        """
        Get a dictionary of key names and their corresponding columns.

        For the given group name, gets the key names and column numbers of the
        dataset and stores them in a dictionary.

        Parameters
        ----------
        file
            hdf5 file object to be read.
        group : str
            Location and name of the dataset. Must contain the full path.

        Returns
        -------
        dict
            Dictionary with keys matching the column names and values matching
            the column indices. Empty dictionary if the location doesn't exist
            or have a 'keys' attribute.

        """
        if group in file and "keys" in file[group].attrs.keys():
            return {self._decode_attr(key): i
                    for i, key in enumerate(file[group].attrs["keys"])}
        else:
            return {}

    @staticmethod
    def _convert_ray_value(ray_value):
        """
        Convert the given ray value into an integer index.

        Allows for common conversion of 'direct', 'reflected', etc. into the
        proper indices.

        Parameters
        ----------
        ray_value : int or str
            Value specifying the ray solution uniquely.

        Returns
        -------
        int
            Integer index corresponding the the given `ray_value`.

        """
        if isinstance(ray_value, str):
            ray_value = ray_value.lower()
            if ray_value == "direct":
                return 0
            elif ray_value == "reflected":
                return 1
            else:
                raise ValueError("Ray/waveform string value '"+ray_value+
                                 "' unsupported")
        elif isinstance(ray_value, (int, float)):
            return int(ray_value)
        else:
            return ray_value



class EventIterator(HDF5Base):
    """
    Class for iterating over event data from an hdf5 file.

    Stores data for a chunk of events according to the `slice_range`.
    Increasing the `slice_range` should result in an improvement in speed,
    while decreasing it should result in an improvement in memory consumption.

    Parameters
    ----------
    hdf5_file
        Open file object to be read from.
    slice_range : int, optional
        Number of events to include in each slice when iterating the file.
    start_event : int, optional
        Event index of the file from which to start iteration.
    stop_event : int, optional
        Event index of the file at which to stop iteration (this event will
        not be included in the iteration).
    step : int, optional
        Step size of the iterator. Must be greater than zero.

    Attributes
    ----------
    triggered
    noise_bases
    is_neutrino
    flavor
    is_nubar

    """
    def __init__(self, hdf5_file, slice_range=None,
                 start_event=None, stop_event=None, step=None):
        self._object = hdf5_file
        HDF5Base.__init__(self, self._object.attrs["version_major"],
                          self._object.attrs["version_minor"])

        # Set locations including non-default locations present in the file
        self._locations_original = self._generate_location_names(
            self._object, self._dataset_locations()
        )
        self._locations = {}
        # Map metadata group datasets
        for key, value in self._locations_original.items():
            if value+"/str" in self._object and value+"/float" in self._object:
                self._locations[key+"_str"] = value+"/str"
                self._locations[key+"_float"] = value+"/float"
            else:
                self._locations[key] = value

        self._max_antenna = 0
        for key in ["antennas_meta_float", "antennas_meta_str"]:
            if not key in self._locations:
                continue
            loc = self._locations[key]
            if loc in self._object:
                self._max_antenna = max(self._max_antenna,
                                        len(self._object[loc]))

        self._max_events = hdf5_file[self._locations["indices"]].shape[0]
        self._total_thrown = hdf5_file[self._locations_original["particles_meta"]].attrs["total_thrown"]

        start_event = 0 if start_event is None else start_event
        step = 1 if step is None else step
        stop_event = self._max_events if stop_event is None else stop_event

        if start_event<0:
            start_event += self._max_events
        if stop_event<0:
            stop_event += self._max_events

        if (start_event<0 or start_event>=self._max_events or
                stop_event<=0 or stop_event>self._max_events):
            raise IndexError("Event index out of range")

        if step<=0:
            raise ValueError("Invalid step size ("+str(step)+")")

        self._slice_range = slice_range
        self._slice_start_event = start_event
        self._slice_end_event = self._slice_start_event
        self._slice_step = step
        self._iter_counter = -1
        self._iter_stop_event = stop_event

        self._data = {}

        self._bool_dict = self._get_bool_dict(self._object, self._locations)

        self._index_keys = [
            self._decode_attr(key)
            for key in self._object[self._locations["indices"]].attrs["keys"]
        ]

        self._keys = {}
        for key, value in self._locations.items():
            self._keys[key] = self._get_keys_dict(self._object, value)

    def _get_index_from_list(self, string_to_comp, keys):
        """
        Get the index of the `string_to_comp` in the list of `keys`.

        Parameters
        ----------
        string_to_comp : str
            String to be matched to one of the keys.
        keys : list of str
            List of keys to be matched.

        Returns
        -------
        int
            Index of the `string_to_comp` in `keys`. -1 if not in `keys`.

        """
        if string_to_comp == self._locations["indices"]:
            raise ValueError("This is the index dataset")
        for i, key in enumerate(keys):
            if string_to_comp in key:
                return i
        return -1

    def __iter__(self):
        return self

    def __next__(self):
        """
        Iterate to the next event in the file.

        Increments the internal iteration counter to use the data for the next
        event. Loads the next slice of data from the file if necessary.

        """
        self._iter_counter += 1
        event_number = (self._iter_counter * self._slice_step
                        + self._slice_start_event)
        if event_number >= self._iter_stop_event:
            raise StopIteration

        if event_number >= self._slice_end_event:
            self._iter_counter = 0
            self._slice_start_event = event_number
            self._slice_end_event = min(self._slice_start_event +
                                        self._slice_range, self._max_events)
            self._load_data()

        return self

    def _load_data(self):
        """
        Load the next slice of data from the file and store it in memory.

        """
        # Store event indices first
        slc = slice(self._slice_start_event,
                    self._slice_end_event,
                    self._slice_step)
        for key, val in self._locations.items():
            if key=="indices":
                continue
            parts = key.split("_")
            if parts[-1]=="float" or parts[-1]=="str":
                parts.pop(-1)
            key_org = "_".join(parts)
            val_org = self._locations_original[key_org]
            index = self._get_index_from_list(val_org, self._index_keys)
            if index>=0:
                self._data[key] = []
                tmp_indices = self._object[self._locations['indices']][slc, index]
                tmp_start = np.min(tmp_indices[:, 0])
                # If multiple events have the same max index, the last one is
                # the one we want (but using argmax would give the first)
                tmp_end_idx = np.where(tmp_indices[:, 0]==np.max(tmp_indices[:, 0]))[0][-1]
                tmp_end = tmp_indices[tmp_end_idx][0] + tmp_indices[tmp_end_idx][1]
                tmp = self._object[val][tmp_start:tmp_end]
                start = 0
                for length in tmp_indices[:, 1]:
                    self._data[key].append(tmp[start:start+length])
                    start = start+length


    def _confirm_iterating(self):
        """
        Convenience function to confirm that iteration is taking place.

        """
        if self._slice_start_event == self._slice_end_event:
            raise ValueError("This function is should be called after "+
                             "initializing the iterator object and calling "+
                             "next(<iterator>)")


    def _get_event_data(self, name):
        """
        Get data from the given dataset or metadata group.

        Parameters
        ----------
        name : str
            Short name of the dataset or metadata group (not the full path).

        Returns
        -------
        ndarray
            If the `name` is a dataset, returns a single array of data.
            If the `name` is a metadata group, returns a tuple of two
            arrays representing the float data and string data, respectively.

        Raises
        ------
        ValueError
            If the dataset is not known by the `EventIterator`.

        """
        if name in self._bool_dict:
            return (np.array([]) if not self._bool_dict[name]
                    else self._data[name][self._iter_counter])
        elif (name+"_float" in self._bool_dict and
              name+"_str" in self._bool_dict):
            float_data = (np.array([]) if not self._bool_dict[name+"_float"]
                          else self._data[name+"_float"][self._iter_counter])
            str_data = (np.array([]) if not self._bool_dict[name+"_str"]
                        else self._data[name+"_str"][self._iter_counter])
            return float_data, str_data
        else:
            raise ValueError("Dataset "+str(name)+" not found")


    def get_data(self, dataset, attribute=None):
        """
        Get data from a given dataset.

        Parameters
        ----------
        dataset : str
            Name or location of the dataset.
        attribute : str
            Attribute of the data to retreive.

        Returns
        -------
        tuple or ndarray
            Event data stored in the `dataset`, optionally narrowed by the
            `attribute`. If `dataset` is actually a group containing float and
            string datasets, and no `attribute` is specified, a tuple of
            ndarray will be returned. Otherwise the return value will be an
            ndarray.

        Raises
        ------
        ValueError
            If the dataset or attribute is unknown or if the dataset has no
            event-specific data.

        """
        if dataset.endswith("_str") or dataset.endswith("_float"):
            name = "_".join(dataset.split("_")[:-1])
        else:
            name = dataset

        if name in self._locations_original:
            location = self._locations_original[name]
        else:
            for key, val in self._locations_original.items():
                if name==val:
                    name = key
                    location = val
                    break
            else:
                analysis_name = self._analysis_location(name)
                for key, val in self._locations_original.items():
                    if analysis_name==val:
                        name = key
                        location = val
                        break
                else:
                    raise ValueError("Dataset "+str(dataset)+" not found")

        if location not in self._index_keys:
            raise ValueError("No event-specific data is available in "+
                             str(dataset))

        data = self._get_event_data(name)

        if isinstance(data, tuple):
            if len(data[0])==0 and len(data[1])==0:
                logger.debug("No relevant data was stored for event %s",
                             self._iter_counter * self._slice_step
                             + self._slice_start_event)
                return data[0]
        else:
            if len(data)==0:
                logger.debug("No relevant data was stored for event %s",
                             self._iter_counter * self._slice_step
                             + self._slice_start_event)
                return data

        if attribute is None:
            if isinstance(data, tuple):
                return data[0], self._decode_string_data(data[1])
            else:
                return data
        elif isinstance(attribute, str):
            if isinstance(data, tuple):
                if attribute in self._keys[name+"_float"]:
                    index = self._keys[name+"_float"][attribute]
                    return data[0][:, index]
                elif attribute in self._keys[name+"_str"]:
                    index = self._keys[name+"_str"][attribute]
                    return self._decode_string_data(data[1][:, index])
                else:
                    raise ValueError("Unrecognized attribute '"+attribute+"'")
            else:
                if attribute in self._keys[name]:
                    index = self._keys[name][attribute]
                    return data[:, index]
                else:
                    raise ValueError("Unrecognized attribute '"+attribute+"'")
        else:
            raise ValueError("Only string values supported as argument")


    def get_waveforms(self, antenna_id=None, waveform_type=None):
        """
        Get waveform data for the event.

        Returns waveform data from the event which can be narrowed by antenna
        and/or waveform type.

        Parameters
        ----------
        antenna_id : int, optional
            Specific antenna number for which to get waveforms.
        waveform_type : int or str, optional
            Specific waveform type (direct / reflected) for which to get
            waveforms.

        Returns
        -------
        ndarray
            Specified waveform data from the event.

        """
        self._confirm_iterating()

        if not self._bool_dict["waveforms"]:
            raise ValueError("Waveform data was not saved in this file")

        data = self._get_event_data("waveforms")
        if len(data)==0:
            logger.debug("No waveform data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)
            return data

        if antenna_id is None:
            antenna_id = slice(None)
        elif isinstance(antenna_id, int) and antenna_id >= self._max_antenna:
            raise ValueError("Antenna Id provided is greater than the number "+
                             "of antennas in detector")

        wf_index = self._convert_ray_value(waveform_type)
        if wf_index is None:
            return data[:, antenna_id]
        elif isinstance(wf_index, int):
            if wf_index>=data.shape[0]:
                return np.array([])
            return data[wf_index, antenna_id]
        else:
            raise ValueError("The parameter 'waveform_type' should either be "+
                             "an integer or a string")


    def get_particle_info(self, attribute=None):
        """
        Get particle data for the event.

        Returns all particle data or a specified `attribute` of the particle
        data.

        Parameters
        ----------
        attribute : str, optional
            Attribute of the particle data to get.

        Returns
        -------
        list or ndarray
            List of dictionaries with all particle data or array of the desired
            attributes for each particle of the event.

        """
        custom_values = ["position", "vertex", "direction", "interaction_info"]
        # Possibly add distance and radius to custom_values
        self._confirm_iterating()

        try:
            data_saved = (self._bool_dict["particles_meta_float"]
                          or self._bool_dict["particles_meta_str"])
        except KeyError:
            data_saved = self._bool_dict["particles_meta"]

        if not data_saved:
            raise ValueError("Particle data was not saved in this file")

        float_data, str_data = self._get_event_data("particles_meta")

        if len(float_data)==0 and len(str_data)==0:
            logger.debug("No particle data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)
            return float_data

        if attribute is None:
            return self._read_metadata_to_dicts(
                data=self._data,
                name="particles_meta",
                index=self._iter_counter,
                str_keys=[item[0] for item in
                          sorted(self._keys['particles_meta_str'].items(),
                                 key=lambda x: x[1])],
                float_keys=[item[0] for item in
                            sorted(self._keys['particles_meta_float'].items(),
                                   key=lambda x: x[1])]
            )

        elif isinstance(attribute,str):
            if attribute in custom_values:
                vector_len = 3
                if attribute == "position" or attribute == "vertex":
                    index = self._keys["particles_meta_float"]["vertex_x"]
                    return float_data[:, index:index+vector_len]
                elif attribute == "direction":
                    index = self._keys["particles_meta_float"]["direction_x"]
                    return float_data[:, index:index+vector_len]
                elif attribute == "interaction_info":
                    groups = ["particles_meta_float", "particles_meta_str"]
                    datasets = [float_data, str_data]
                    dic = {}
                    for key, value in self._keys["particles_meta_float"].items():
                        if "interaction" in key:
                            dic[key] = float_data[:, value]
                    for key, value in self._keys["particles_meta_str"].items():
                        if "interaction" in key:
                            dic[key] = self._decode_string_data(str_data[:, value])
                    return dic
                else:
                    raise ValueError("This value is not supported yet")
            else:
                if attribute in self._keys["particles_meta_float"]:
                    index = self._keys["particles_meta_float"][attribute]
                    return float_data[:, index]
                elif attribute in self._keys["particles_meta_str"]:
                    index = self._keys["particles_meta_str"][attribute]
                    return self._decode_string_data(str_data[:, index])
                else:
                    raise ValueError("Unrecognized particle attribute '"+
                                     attribute+"'")

        else:
            raise ValueError("Only string values supported as argument")


    def get_rays_info(self, attribute=None):
        """
        Get ray path data for the event.

        Returns all ray path data or a specified `attribute` of the data.

        Parameters
        ----------
        attribute : str, optional
            Attribute of the ray path data to get.

        Returns
        -------
        list or ndarray
            List of dictionaries with all ray path data or array of the desired
            attributes for each ray of the event.

        """
        custom_values = ["polarization", "emitted_direction",
                         "received_direction"]
        self._confirm_iterating()

        try:
            data_saved = (self._bool_dict["rays_meta_float"]
                          or self._bool_dict["rays_meta_str"])
        except KeyError:
            data_saved = self._bool_dict["rays_meta"]

        if not data_saved:
            raise ValueError("Ray data was not saved in this file")

        float_data, str_data = self._get_event_data("rays_meta")

        if len(float_data)==0 and len(str_data)==0:
            logger.debug("No ray data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)
            return float_data

        if attribute is None:
            return self._read_metadata_to_dicts(
                data=self._data,
                name="rays_meta",
                index=self._iter_counter,
                str_keys=[item[0] for item in
                          sorted(self._keys['rays_meta_str'].items(),
                                 key=lambda x: x[1])],
                float_keys=[item[0] for item in
                            sorted(self._keys['rays_meta_float'].items(),
                                   key=lambda x: x[1])]
            )

        elif isinstance(attribute,str):
            if attribute in custom_values:
                if attribute == "polarization":
                    index = self._keys["rays_meta_float"]["polarization_x"]
                elif attribute == "emitted_direction":
                    index = self._keys["rays_meta_float"]["emitted_x"]
                elif attribute == "received_direction":
                    index = self._keys["rays_meta_float"]["received_x"]
                else:
                    raise ValueError("The attribute is not supported yet")
                return float_data[:, :, index:index+3]

            else:
                if attribute in self._keys["rays_meta_float"]:
                    index = self._keys["rays_meta_float"][attribute]
                    return float_data[:, :, index]
                elif attribute in self._keys["rays_meta_str"]:
                    index = self._keys["rays_meta_str"][attribute]
                    return self._decode_string_data(str_data[:, :, index])
                else:
                    raise ValueError("Unrecognized particle attribute '"+
                                     attribute+"'")

        else:
            raise ValueError("Only string values supported as argument")

    @property
    def triggered(self):
        """
        Whether or not the event triggered the detector.

        `None` if no trigger data is available for the event.

        """
        self._confirm_iterating()
        if not self._bool_dict["triggers"]:
            raise ValueError("Trigger data was not saved in this file")
        triggered = self._get_event_data("triggers")
        if len(triggered)==0:
            return None
        else:
            return triggered[0]

    @property
    def noise_bases(self):
        """
        Noise bases for each antenna of the event for reproducing noise.

        """
        self._confirm_iterating()
        if not self._bool_dict["noise"]:
            raise ValueError("Noise data was not saved in this file")
        bases = self._get_event_data("noise")
        if len(bases)==0:
            logger.debug("No noise data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)
            return bases
        else:
            return bases[0]

    def get_triggered_components(self, ray=None):
        """
        Get the components of the detector triggered by the event.

        Returns a list of the triggered component names for the event as a
        whole or for a specific ray path.

        Parameters
        ----------
        ray : int or str, optional
            Specific ray path (direct / reflected) for which to get the
            triggered components.

        Returns
        -------
        list of str
            List of the names of the components triggered by the event.

        """
        self._confirm_iterating()
        if not self._bool_dict["mc_triggers"]:
            raise ValueError("Monte Carlo trigger data was not saved in this "+
                             "file")
        triggers = self._get_event_data("mc_triggers")
        if len(triggers)==0:
            logger.debug("No monte carlo trigger data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)

        ray_number = self._convert_ray_value(ray)
        if ray_number is not None:
            if triggers.ndim==1:
                triggers = np.array([triggers])
            if ray_number>=triggers.shape[0]:
                return []
            else:
                triggers = triggers[ray_number]

        if triggers.ndim!=1:
            triggers = np.any(triggers, axis=0)

        return [key for key, val in self._keys["mc_triggers"].items()
                if triggers[val]]


    @property
    def flavor(self):
        """
        Flavor of the event's initial particle.

        If the initial particle is not a neutrino, returns a null string.

        """
        name = self.get_particle_info("particle_name")[0]
        if "neutrino" in name:
            # Grabbing the first part of the name, that is the flavor of the
            # particle
            return name.split("_")[0]
        else:
            # Returning null if the particle is not neutrino
            return ""

    @property
    def is_nubar(self):
        """
        Whether the event's initial particle is an anti-neutrino.

        `None` if the initial particle is not a neutrino.

        """
        if self.is_neutrino:
            return self.get_particle_info("particle_id")[0] < 0

    @property
    def is_neutrino(self):
        """
        Whether the event's initial particle is a neutrino.

        """
        return "neutrino" in self.get_particle_info("particle_name")[0]

    @property
    def total_events_thrown(self):
        """
        Proportional number of events thrown by the generator.

        Uses the total number of events thrown in the full file and portions
        them evenly among the actual events to give the total number of
        events thrown up to the present event.

        """
        event_number = (self._iter_counter * self._slice_step
                        + self._slice_start_event)
        return int((event_number+1)/self._max_events*self._total_thrown)



class HDF5Reader(BaseReader, HDF5Base):
    """
    Class for reading data from an hdf5 file.

    Works as a context manager and allows for reading simulation/real data or
    analysis-level data from the given file.

    Parameters
    ----------
    filename : str
        File name to open in read mode.
    slice_range : int, optional
        Number of events to include in each slice when iterating the file.
        Loads the entire file at once by default. For large files specifying
        a value will result in the file being read in chunks, allowing for
        reduced memory consumption at the cost of some speed.

    Attributes
    ----------
    filename : str
        Name of the file (to be) opened.
    is_open
    antenna_info
    file_metadata

    See Also
    --------
    pyrex.io.EventIterator : Class for iterating over event data from an hdf5
                             file.

    """
    def __init__(self, filename, slice_range=None):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
            self._slice_range = slice_range
        else:
            raise ValueError(filename+" is not in the HDF5 format")
        self._is_open = False

    def __getitem__(self, key):
        """
        Get a specified item from the file.

        Supports access to slices or individual events, as well as direct
        access to the hdf5 groups and datasets.

        Parameters
        ----------
        key : int or slice or str
            If `int`, gets a single event. If `slice`, gets a slice of events.
            If `str`, accesses the corresponding hdf5 group or dataset.

        Returns
        -------
        EventIterator or Group or Dataset
            `EventIterator` if a `key` is an integer or slice. `Group` or
            `Dataset` object of the hdf5 file if `key` is a string.

        """
        if not self.is_open:
            raise IOError("File is not open")
        if isinstance(key, int):
            stop = self._num_events if key==-1 else key+1
            it = EventIterator(hdf5_file=self._file,
                               slice_range=1,
                               start_event=key,
                               stop_event=stop,
                               step=1)
            return next(it)
        elif isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = self._num_events if key.stop is None else key.stop
            slice_range = min(self._slice_range, stop-start)
            return EventIterator(hdf5_file=self._file,
                                 slice_range=slice_range,
                                 start_event=key.start,
                                 stop_event=key.stop,
                                 step=key.step)
        elif isinstance(key, str):
            loc = self._get_key_location(key)
            if loc in self._file:
                return self._file[loc]
            else:
                raise ValueError("'"+key+"' does not exist in "
                                 +str(self.filename))

        else:
            raise ValueError("Invalid key '"+str(key)+"'")

    def __contains__(self, item):
        """
        Check whether the given `item` is in the reader.

        Only supports checks for groups or datasets in the wrapped hdf5 file
        object.

        Parameters
        ----------
        item : str
            The hdf5 group or dataset to be checked for existence.

        Returns
        -------
        bool
            Existence of the `item` in the reader file.

        """
        if not self.is_open:
            raise IOError("File is not open")
        return self._get_key_location(item) in self._file

    def __len__(self):
        """
        Length of the file (i.e. the number of events stored).

        """
        if not self.is_open:
            raise IOError("File is not open")
        return self._num_events

    def __iter__(self):
        """
        Iterable responsible for returning each event in turn.

        """
        if not self.is_open:
            raise IOError("File is not open")
        return EventIterator(hdf5_file=self._file,
                             slice_range=self._slice_range)

    def _get_key_location(self, key):
        """
        Get the location of a key representing a group or dataset in the file.

        Parameters
        ----------
        key : str
            Key representing the name or full path of a group or dataset to be
            found in the file.

        Returns
        -------
        str
            Full path of the group or dataset in the file. Null string if the
            group or dataset doesn't exist in the file.

        """
        if key in self._file:
            return key
        elif key in self._locations_original:
            return self._locations_original[key]
        elif key in self._locations:
            return self._locations[key]
        elif self._analysis_location(key) in self._file:
            return self._analysis_location(key)
        else:
            return ""

    def open(self):
        """
        Open the hdf5 file for reading.

        Opens the file in read-only mode and stores some basic information for
        working with the file.

        """
        self._file = h5py.File(self.filename, mode='r')
        self._is_open = True
        HDF5Base.__init__(self, self._file.attrs["version_major"],
                          self._file.attrs["version_minor"] )
        # Set locations including non-default locations present in the file
        self._locations_original = self._generate_location_names(
            self._file, self._dataset_locations()
        )
        self._locations = {}
        # Map metadata group datasets
        for key, value in self._locations_original.items():
            if value+"/str" in self._file and value+"/float" in self._file:
                self._locations[key+"_str"] = value+"/str"
                self._locations[key+"_float"] = value+"/float"
            else:
                self._locations[key] = value

        self._num_ant = 0
        for key in ["antennas_meta_float", "antennas_meta_str"]:
            if not key in self._locations:
                continue
            loc = self._locations[key]
            if loc in self._file:
                self._num_ant = max(self._num_ant, len(self._file[loc]))
        self._iter_counter = None
        # Assuming that the data indices will have all the list of all events
        self._num_events = 0
        if self._locations["indices"] in self._file:
            self._num_events = len(self._file[self._locations["indices"]])

        if self._slice_range is None:
            self._slice_range = self._num_events

        self._bool_dict = self._get_bool_dict(self._file, self._locations)

        self._event_indices_key = self._get_keys_dict(self._file,
                                                      self._locations["indices"])

    def close(self):
        """
        Close the hdf5 file.

        """
        self._file.close()
        self._is_open = False

    @property
    def is_open(self):
        """
        Boolean of whether the file is currently open.

        """
        return self._is_open

    def _get_table_slice(self, group, event_index):
        """
        Get the slice of the group for the given event index.

        Generates the appropriate `slice` object for where the data for the
        event given by `event_index` is stored.

        Parameters
        ----------
        group : str
            Location and name of the dataset or metadata group. Must contain
            the full path.
        event_index : int
            Index of the event whose data slice should be given.

        Returns
        -------
        slice
            Slice object representing the rows of the group which contain data
            for the given event index.

        """
        start = self._file[self._locations["indices"]][event_index,
                                                       self._event_indices_key[group],
                                                       0]
        length = self._file[self._locations["indices"]][event_index,
                                                        self._event_indices_key[group],
                                                        1]
        return slice(start, start+length)


    def _get_waveform_of_one_kind(self, wf_type):
        """
        Get all waveforms of a single type.

        Convenience function for returning all waveforms with the given type
        in the file. Unfortunately requires iteration of the entire file due to
        the way waveforms are stored. Should be used with caution.

        Parameters
        ----------
        wf_type : int or str
            Waveform type to collect.

        Returns
        -------
        ndarray
            Array of all waveforms of the given type in the file.

        """
        logger.warning("Getting all waveforms of a single type requires "+
                       "iteration over the entire dataset. Consider iterating "+
                       "the file manually instead.")
        return np.asarray(
            [event.get_waveforms(waveform_type=wf_type) for event in self]
        )

    def get_waveforms(self, event_id=None, antenna_id=None, waveform_type=None):
        """
        Get waveform data from the file.

        Capable of retreiving waveform data per event, per antenna, per
        waveform type (direct / reflected), or any combination of these.

        Parameters
        ----------
        event_id : int, optional
            Specific event id for which to get waveforms.
        antenna_id : int, optional
            Specific antenna number for which to get waveforms.
        waveform_type : int or str, optional
            Specific waveform type (direct / reflected) for which to get
            waveforms.

        Returns
        -------
        ndarray
            Waveform data from the specified portions of the file.

        """
        if not self.is_open:
            raise IOError("File is not open")

        if not self._bool_dict["waveforms"]:
            raise ValueError("Waveform data was not saved in this file")

        if event_id is None and waveform_type is not None:
            return self._get_waveform_of_one_kind(waveform_type)

        if event_id is None:
            event_id = slice(None)
        else:
            event_id = self._get_table_slice(
                self._locations["waveforms"], event_id
            )

        if antenna_id is None:
            antenna_id = slice(None)
        elif isinstance(antenna_id, int) and antenna_id >= self._num_ant:
            raise ValueError("Antenna Id provided is greater than the number "+
                             "of antennas in detector")

        wf_data = self._file[self._locations["waveforms"]]

        wf_index = self._convert_ray_value(waveform_type)
        if wf_index is None:
            return wf_data[event_id, antenna_id]
        elif isinstance(wf_index, int):
            start = event_id.start
            stop = event_id.stop
            if start is None:
                start = 0
            if stop is None:
                stop = wf_data.shape[0]
            if wf_index>=stop-start:
                raise ValueError("Invalid waveform index ("+str(wf_index)+")")
            return wf_data[start+wf_index, antenna_id]
        else:
            raise ValueError("The parameter 'waveform_type' should either be "+
                             "an integer or a string")

    @classmethod
    def _recursive_combine_dicts(cls, dicts1, dicts2):
        """
        Recursively combine (lists of) dictionaries into a single (list of)
        dictionaries.

        If given two dictionaries, returns the combined dictionary (with keys
        of `dicts2` taking precedence). If given two lists (of lists...) of
        dictionaries, assuming the lists have the same dimensionality will
        return a list with the same dimensionality where the dictionaries of
        the same positions have been combined.

        Parameters
        ----------
        dicts1, dicts2 : dict or list
            (Lists of) dictionaries to combine.

        Returns
        -------
        dict or list
            (List of) combined dictionaries.

        """
        if isinstance(dicts1, dict) and isinstance(dicts2, dict):
            dicts1.update(dicts2)
            return dicts1
        elif (isinstance(dicts1, list) and isinstance(dicts2, list) and
              len(dicts1)==len(dicts2)):
            return [cls._recursive_combine_dicts(d1, d2)
                    for d1, d2 in zip(dicts1, dicts2)]
        else:
            raise ValueError("Unsupported types")

    @property
    def antenna_info(self):
        """
        Antenna data from the file.

        List of dictionaries containing all available antenna data, both real
        and monte-carlo based.

        """
        if not self.is_open:
            raise IOError("File is not open")
        if (not self._bool_dict["antennas"] and
                not self._bool_dict["antennas_meta_float"] and
                not self._bool_dict["antennas_meta_str"]):
            raise ValueError("Antenna data was not saved in this file")
        ant_dict_data = self._read_dataset_to_dicts(
            self._file, self._locations_original["antennas"]
        )
        ant_dict_MC = self._read_metadata_to_dicts(
            self._file, self._locations_original["antennas_meta"]
        )
        return self._recursive_combine_dicts(ant_dict_data, ant_dict_MC)

    @property
    def file_metadata(self):
        """
        Metadata from the file's metadata datasets.

        """
        if not self.is_open:
            raise IOError("File is not open")
        if (not self._bool_dict["file_meta_float"] and
                not self._bool_dict["file_meta_str"]):
            raise ValueError("File metadata was not saved in this file")
        return self._read_metadata_to_dicts(
            self._file, self._locations_original["file_meta"]
        )

    @property
    def total_events_thrown(self):
        """
        Total number of events thrown by the generator.

        This value may be different than the number of events in the file,
        particularly for shadowing generators which skip events that don't
        survive travel through the Earth.

        """
        if not self.is_open:
            raise IOError("File is not open")
        return self._file[self._locations_original['particles_meta']].attrs['total_thrown']




class HDF5Writer(BaseWriter, HDF5Base):
    """
    Class for writing data to an hdf5 file.

    Works as a context manager and allows for writing simulation/real data or
    analysis-level data to the given file.

    Parameters
    ----------
    filename : str
        File name to open in the given write mode.
    mode : str, optional
        Mode with which to open the file. 'w' writes to the file, overwriting
        any existing data. 'x' writes to the file, failing if the file exists.
        'a' appends to the file, creating if the file doesn't exist. 'r+'
        appends to the file, failing if the file doesn't exist.
    write_particles : bool, optional
        Whether to write particle metadata when adding event data to the file.
    write_waveforms : bool, optional
        Whether to write waveform data when adding event data to the file.
    write_triggers : bool, optional
        Whether to write trigger data when adding event data to the file.
    require_trigger : bool or list of str, optional
        Whether to write data only when the detector is triggered. If ``False``
        all data will be written for every event. If ``True`` most data will
        be written only on a detector trigger (particle metadata and trigger
        data will still be written for every event). If a list of strings is
        provided, then the specified data types will be written only on a
        detector trigger and all other data types will be written for every
        event.

    Attributes
    ----------
    filename : str
        Name of the file (to be) opened.
    is_open
    has_detector

    Other Parameters
    ----------------
    write_antenna_triggers : bool, optional
        Whether to write trigger data of individual antennas when adding event
        data to the file.
    write_rays : bool, optional
        Whether to write ray metadata when adding event data to the file.
    write_noise : bool, optional
        Whether to write noise-generation metadata when adding event data to
        the file.

    """
    def __init__(self, filename, mode='x', write_particles=True,
                 write_triggers=True, write_antenna_triggers=False,
                 write_rays=True, write_noise=False, write_waveforms=False,
                 require_trigger=True):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            self.filename = filename+".h5"
        if mode not in ['w', 'x', 'a', 'r+']:
            raise ValueError("Unrecognized file mode '"+str(mode)+"'")
        self._mode = mode
        self._is_open = False

        # Set file version
        if H5PY_V2_STRINGS:
            HDF5Base.__init__(self, 1, 0)
        else:
            HDF5Base.__init__(self, 1, 1)
        self._data_locs = self._dataset_locations()

        if write_antenna_triggers and not write_triggers:
            raise ValueError("A true value for 'write_antenna_triggers' "+
                             "requires a true value for 'write_triggers'")

        self._write_data = {
            "particles": write_particles,
            "triggers": write_triggers,
            "antenna_triggers": write_antenna_triggers,
            "waveforms": write_waveforms,
            "rays": write_rays,
            "noise": write_noise
        }

        if isinstance(require_trigger, bool):
            self._trig_only = {key: require_trigger for key in self._write_data}
            if require_trigger:
                always_write = ["particles", "triggers", "antenna_triggers"]
                self._update_bool_dict(self._trig_only, always_write, False)
        else:
            self._trig_only = {key: False for key in self._write_data}
            self._update_bool_dict(self._trig_only, require_trigger, True)


    @staticmethod
    def _update_bool_dict(dictionary, keys, value):
        """
        Sets the `keys` of `dictionary` to `value`.

        Parameters
        ----------
        dictionary : dict
            Dictionary to be updated.
        keys : str or list of str
            Key(s) of `dictionary` which should take on the given `value`.
        value
            Value to store in the `keys` of `dictionary`.

        """
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, (list, tuple)):
            for key in keys:
                if key=="":
                    continue
                if key not in dictionary:
                    raise ValueError("Key '"+key+"' not recognized")
                else:
                    dictionary[key] = value
        else:
            raise TypeError("Unrecognized type for keys")

    def __getitem__(self, key):
        """
        Get a specified group or dataset from the file.

        Provides direct access to the hdf5 groups and datasets.

        Parameters
        ----------
        key : str
            The hdf5 group or dataset to be given.

        Returns
        -------
        Group or Dataset
            Group or dataset object of the hdf5 file.

        """
        if isinstance(key, str):
            loc = self._get_key_location(key)
            if loc in self._file:
                return self._file[loc]
            else:
                raise ValueError("'"+key+"' does not exist in "
                                 +str(self.filename))
        else:
            raise ValueError("Invalid key '"+str(key)+"'")

    def __contains__(self, item):
        """
        Check whether the given `item` is in the writer.

        Only supports checks for groups or datasets in the wrapped hdf5 file
        object.

        Parameters
        ----------
        item : str
            The hdf5 group or dataset to be checked for existence.

        Returns
        -------
        bool
            Existence of the `item` in the writer file.

        """
        return self._get_key_location(item) in self._file

    def __delitem__(self, key):
        """
        Delete a specified group or dataset from the file.

        Provides direct access to delete the hdf5 groups and datasets.

        Parameters
        ----------
        key : str
            The hdf5 group or dataset to be removed.

        """
        if isinstance(key, str):
            loc = self._get_key_location(key)
            del self._file[loc]
        else:
            raise ValueError("Invalid key '"+str(key)+"'")

    def _get_key_location(self, key):
        """
        Get the location of a key representing a group or dataset in the file.

        Parameters
        ----------
        key : str
            Key representing the name or full path of a group or dataset to be
            found in the file.

        Returns
        -------
        str
            Full path of the group or dataset in the file. Null string if the
            group or dataset doesn't exist in the file.

        """
        if not self.is_open:
            raise IOError("File is not open")
        if key in self._file:
            return key
        elif key in self._data_locs:
            return self._data_locs[key]
        elif key.endswith('_str') and key[:-4] in self._data_locs:
            return self._data_locs[key[:-4]]+'/str'
        elif key.endswith('_float') and key[:-6] in self._data_locs:
            return self._data_locs[key[:-6]]+'/float'
        elif self._analysis_location(key) in self._file:
            return self._analysis_location(key)
        else:
            return ""

    def open(self):
        """
        Open the hdf5 file for writing.

        Opens the file in the mode given on initialization and records basic
        file-level metadata.

        """
        # Create empty file
        self._file = h5py.File(self.filename, mode=self._mode)
        self._is_open = True

        # Special append-mode opening method
        if ((self._mode=='a' or self._mode=='r+') and
                self._data_locs['file_meta'] in self._file):
            # Set file version
            HDF5Base.__init__(self, self._file.attrs['version_major'],
                              self._file.attrs['version_minor'])
            self._data_locs = self._dataset_locations()
            # Set event number counters
            self._counters = {}
            for key, val in self._data_locs.items():
                if key in ["file_meta", "antennas", "antennas_meta"]:
                    continue
                if val not in self._file:
                    self._counters[key] = 0
                else:
                    if key.endswith("meta"):
                        count = 0
                        for table in ["float", "str"]:
                            loc = val+"/"+table
                            if loc in self._file:
                                count = max(count, self._file[loc].shape[0])
                    else:
                        count = self._file[val].shape[0]
                    self._counters[key] = count
            return

        self._create_dataset(self._data_locs['indices'])
        self._file.attrs['version_major'] = self._file_version_major
        self._file.attrs['version_minor'] = self._file_version_minor

        # Set event number counters
        self._counters = {key: 0 for key in self._data_locs
                          if key not in ["file_meta", "antennas",
                                         "antennas_meta"]}

        # Write some generic metadata about the file production
        self._create_metadataset(self._data_locs['file_meta'])
        major, minor, patch = __version__.split('.')
        now = datetime.datetime.now()
        i = 0
        stack = inspect.stack()
        for i, frame in enumerate(stack):
            if frame.function == "open":
                break
        if i < len(stack) and stack[i+1].function == "__enter__":
            if i+1 < len(stack):
                opening_script = stack[i+2].filename
            else:
                opening_script = stack[i+1].filename
        else:
            opening_script = stack[i+1].filename
        metadata = {
            "file_version": (str(self._file_version_major)+"."+
                             str(self._file_version_minor)),
            "file_version_major": self._file_version_major,
            "file_version_minor": self._file_version_minor,
            "pyrex_version": __version__,
            "pyrex_version_major": int(major),
            "pyrex_version_minor": int(minor),
            "pyrex_version_patch": int(patch),
            "opening_script": opening_script,
            "top_level_script": stack[-1].filename,
            "datetime": now.strftime('%Y-%d-%m %H:%M:%S'),
            "date": now.strftime('%Y-%d-%m'),
            "time": now.strftime('%H:%M:%S'),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
        }
        self._write_metadata(self._data_locs['file_meta'], metadata)

    def close(self):
        """
        Close the hdf5 file.

        """
        self._file.close()
        self._is_open = False

    @property
    def is_open(self):
        """
        Boolean of whether the file is currently open.

        """
        return self._is_open


    def _create_dataset(self, name):
        """
        Create the given dataset in the file.

        Creates datasets of different shapes and sizes based on the rules for
        expected datasets in the file. If the given dataset already exists,
        will simply return the existing dataset rather than recreating it.

        Parameters
        ----------
        name : str
            Location and name of the dataset to be created. Must contain the
            full path to the dataset location and the name of the dataset.

        Returns
        -------
        dataset
            The dataset object corresponding to the created (or existing)
            dataset requested.

        Raises
        ------
        ValueError
            If there is no rule for building the given dataset.

        """
        if not self.is_open:
            raise IOError("File is not open")

        if not name.startswith("/"):
            name = "/"+name

        # Don't recreate datasets
        if name in self._file:
            return self._file[name]

        if name==self._data_locs['indices']:
            data = self._file.create_dataset(
                name=name, shape=(0, 0, 2),
                dtype=np.int_, maxshape=(None, None, 2)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "tables"
            data.dims[2].label = "indices"
            data.attrs['keys'] = []
            return

        elif name==self._data_locs['waveforms']:
            # Dimension lengths:
            #   0 - Total number of waveforms across all events
            #   1 - Number of antennas
            #   2 - Number of value types (2: times & values)
            data = self._file.create_dataset(
                name=name, shape=(0, len(self._detector), 2),
                dtype=h5py.special_dtype(vlen=np.float_),
                maxshape=(None, len(self._detector), 2)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "antennas"
            data.dims[2].label = "waveforms"

        elif name==self._data_locs['triggers']:
            data = self._file.create_dataset(
                name=name, shape=(0,),
                dtype=np.bool_, maxshape=(None,)
            )
            data.dims[0].label = "events"

        elif name==self._data_locs['antennas']:
            data = self._file.create_dataset(
                name=name, shape=(len(self._detector), 0),
                dtype=np.float_, maxshape=(len(self._detector), None)
            )
            data.dims[0].label = "antennas"
            data.dims[1].label = "attributes"
            keys = [
                "position_x", "position_y", "position_z",
                "z_axis_x", "z_axis_y", "z_axis_z",
                "x_axis_x", "x_axis_y", "x_axis_z"
            ]
            data.attrs['keys'] = [self._encode_attr(key) for key in keys]
            data.resize(len(data.attrs['keys']), axis=1)

        elif name==self._data_locs['mc_triggers']:
            data = self._file.create_dataset(
                name=name, shape=(0, 0),
                dtype=np.bool_, maxshape=(None, None)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "types"
            data.attrs['keys'] = []

        elif name==self._data_locs['noise']:
            data = self._file.create_dataset(
                name=name, shape=(0, len(self._detector), 3),
                dtype=h5py.special_dtype(vlen=np.float_),
                maxshape=(None, len(self._detector), 3)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "antennas"
            data.dims[2].label = "attributes"
            keys = ["frequency", "amplitude", "phase"]
            data.attrs['keys'] = [self._encode_attr(key) for key in keys]
            data.resize(len(data.attrs['keys']), axis=2)

        else:
            raise ValueError("Unrecognized dataset name '"+name+"'")

        return data


    def _create_metadataset(self, name, shape=None, maxshape=None):
        """
        Create the given metadata group in the file.

        Creates metadata groups with 'float' and 'str' datasets of different
        shapes and sizes based on the rules for expected datasets in the file.
        If the given metadata group already exists, will simply return the
        existing group rather than recreating it.

        Parameters
        ----------
        name : str
            Location and name of the metadata group to be created. Must contain
            the full path to the group location and the name of the metadata
            group.
        shape : tuple of int, optional
            Shape of the metadata datasets to be created for analysis metadata
            (not including the attribute dimension). Attribute dimension only
            by default.
        maxshape : tuple of int, optional
            Maxshape of the metadata datasets to be created for analysis
            metadata (not including the attribute dimension). No limits by
            default.

        Returns
        -------
        group
            The group object corresponding to the created (or existing)
            metadata group requested (containing 'float' and 'str' datasets).

        Raises
        ------
        ValueError
            If there is no rule for building the given metadata group.

        """
        if not self.is_open:
            raise IOError("File is not open")

        if not name.startswith("/"):
            name = "/"+name

        # Don't recreate metadatasets
        if name in self._file:
            return self._file[name]

        if name==self._data_locs['file_meta']:
            shape = (0,)
            maxshape = (None,)
            def apply_labels(data):
                data.dims[0].label = "attributes"

        elif name==self._data_locs['particles_meta']:
            shape = (0, 0)
            maxshape = (None, None)
            def apply_labels(data):
                data.dims[0].label = "particles"
                data.dims[1].label = "attributes"

        elif name==self._data_locs['antennas_meta']:
            shape = (len(self._detector), 0)
            maxshape = (len(self._detector), None)
            def apply_labels(data):
                data.dims[0].label = "antennas"
                data.dims[1].label = "attributes"

        elif name==self._data_locs['rays_meta']:
            shape = (0, len(self._detector), 0)
            maxshape = (None, len(self._detector), None)
            def apply_labels(data):
                data.dims[0].label = "events"
                data.dims[1].label = "antennas"
                data.dims[2].label = "attributes"

        # Special case for analysis metadatasets
        elif name.split("/")[2]=="analysis":
            if shape is None:
                shape = (0,)
            else:
                shape = tuple(list(shape)+[0])
            if maxshape is None:
                maxshape = tuple(None for dimension in shape)
            else:
                maxshape = tuple(list(maxshape)+[None])
                if len(maxshape)!=len(shape):
                    raise ValueError("Length of 'maxshape' must match 'shape'")
            apply_labels = lambda data: None

        else:
            raise ValueError("Unrecognized metadataset name '"+name+"'")

        # Create group with the given name, create string and float datasets,
        # link attribute names, and label the dimensions
        named_group = self._file.create_group(name)
        str_data = named_group.create_dataset(
            name="str", shape=shape,
            dtype=h5py.special_dtype(vlen=str), maxshape=maxshape
        )
        str_data.attrs['keys'] = []
        apply_labels(str_data)

        float_data = named_group.create_dataset(
            name="float", shape=shape,
            dtype=np.float_, maxshape=maxshape
        )
        float_data.attrs['keys'] = []
        apply_labels(float_data)

        return named_group


    def _write_metadata(self, name, metadata, index=None):
        """
        Writes the given `metadata` to the metadata group at `name`.

        Splits information in the `metadata` into string and float values and
        stores them in the appropriate datasets of the metadata group. Creates
        new attribute columns for attributes which have not yet been written.

        Parameters
        ----------
        name : str
            Location and name of the metadata group to be written to. Must
            contain the full path to the group location and the name of the
            metadata group.
        metadata : dict or list of dict
            Dictionaries containing attribute column names as keys and float or
            string data as values.
        index : int, optional
            Additional index for specifying the row where metadata should be
            written, used by some higher-dimension metadata groups.

        Raises
        ------
        ValueError
            If `name` is not an existing metadata group, or there is no rule
            for writing to the metadata group.

        """
        if not self.is_open:
            raise IOError("File is not open")

        if not name.startswith("/"):
            name = "/"+name

        if name not in self._file:
            raise ValueError("Metadataset '"+name+"' does not exist")
        elif 'str' not in self._file[name] or 'float' not in self._file[name]:
            raise ValueError("'"+name+"' is not a metadata group")
        str_data = self._file[name]['str']
        float_data = self._file[name]['float']

        if name==self._data_locs['file_meta']:
            data_axis = 0
            def write_value(value, dataset, *indices):
                dataset[indices[0]] = value

        elif name==self._data_locs['particles_meta']:
            data_axis = 1
            def write_value(value, dataset, *indices):
                dataset[indices[1]+indices[2], indices[0]] = value

        elif name==self._data_locs['antennas_meta']:
            data_axis = 1
            def write_value(value, dataset, *indices):
                dataset[indices[1], indices[0]] = value

        elif name==self._data_locs['rays_meta']:
            data_axis = 2
            def write_value(value, dataset, *indices):
                dataset[indices[2], indices[1], indices[0]] = value

        # Special case for analysis metadatasets
        elif name.split("/")[2]=="analysis":
            data_axis = str_data.ndim-1
            def write_value(value, dataset, *indices):
                if dataset.ndim==1:
                    dataset[indices[0]] = value
                elif dataset.ndim==2:
                    dataset[indices[1], indices[0]] = value
                else:
                    dataset[indices[2], indices[1], indices[0]] = value

        else:
            raise ValueError("Unrecognized metadataset name '"+name+"'")

        if isinstance(metadata, dict):
            metadata = [metadata]
        # Write metadata values to the appropriate datasets
        for i, meta in enumerate(metadata):
            for key, val in meta.items():
                if isinstance(val, str):
                    val_type = "string"
                else:
                    try:
                        val_length = len(val)
                    except TypeError:
                        val_length = -1
                        val = [val]
                    if val_length == 0:
                        continue
                    if isinstance(val[0], str):
                        val_type = "string"
                    elif np.isscalar(val[0]):
                        val_type = "float"
                    else:
                        raise ValueError("'"+key+"' value ("+str(val[0]) +
                                         ") must be string or scalar")
                    if val_length == -1:
                        val = val[0]

                if val_type=="string":
                    for j, match in enumerate(str_data.attrs['keys']):
                        if self._decode_attr(match)==key:
                            break
                    else:
                        # If no key matched, add key and resize dataset
                        j = len(str_data.attrs['keys'])
                        str_data.attrs['keys'] = (list(str_data.attrs['keys'])
                                                  + [self._encode_attr(key)])
                        str_data.resize(j+1, axis=data_axis)
                    write_value(val, str_data, j, i, index)
                elif val_type=="float":
                    for j, match in enumerate(float_data.attrs['keys']):
                        if self._decode_attr(match)==key:
                            break
                    else:
                        # If no key matched, add key and resize dataset
                        j = len(float_data.attrs['keys'])
                        float_data.attrs['keys'] = (list(float_data.attrs['keys'])
                                                    + [self._encode_attr(key)])
                        float_data.resize(j+1, axis=data_axis)
                    write_value(val, float_data, j, i, index)
                else:
                    raise ValueError("Unrecognized val_type")


    def _write_indices(self, full_name, start_index, length=1,
                       global_index_value=None):
        """
        Write the given start and length indices to the event indices dataset.

        Parameters
        ----------
        full_name : str
            Location and name of the dataset or metadata group to be indexed.
            Must contain the full path to the location and the name of the
            dataset or metadata group.
        start_index : int
            Starting index in the dataset(s) containing information on the
            event.
        length : int, optional
            Number of rows in the dataset(s) containing data for the event.
        global_index_value : int, optional
            Global event index (row of the event indices dataset) to write to.
            Defaults to the internal event counter of the writer.

        """
        if not self.is_open:
            raise IOError("File is not open")
        if global_index_value is None:
            global_index_value = self._counters['indices']
        indices = self._file[self._data_locs['indices']]
        # Don't write indices if the matching dataset doesn't exist
        if full_name not in self._file:
            return
        encoded_name = self._encode_attr(full_name)
        # Add column for full_name if it doesn't exist
        if (len(indices.attrs['keys'])==0 or
                encoded_name not in indices.attrs['keys']):
            indices.attrs['keys'] = list(indices.attrs['keys']) + [encoded_name]
            indices.resize(indices.shape[1]+1, axis=1)
        for i, key in enumerate(indices.attrs['keys']):
            if key==encoded_name:
                if indices.shape[0]<=global_index_value:
                    indices.resize(global_index_value+1, axis=0)
                indices[global_index_value, i] = (start_index, length)
                return
        raise ValueError("Unrecognized table name '"+full_name+"'")


    def _preset_all_indices(self):
        """
        Set the indices for the current event based on the counter.

        Sets each index to the value of the counter with a length of zero. This
        should be overwritten by the true index and length when that is
        determined.

        """
        for key, count in self._counters.items():
            if key=="indices":
                continue
            self._write_indices(self._data_locs[key], count, 0)


    def set_detector(self, detector):
        """
        Link the given `detector` to the file.

        This `detector` is used for writing antenna data and metadata, as well
        as for reading in waveforms and other data. Typically will be the
        same detector object used in the `EventKernel`.

        """
        if not self.is_open:
            raise IOError("File is not open")
        self._detector = detector
        data = self._create_dataset(self._data_locs['antennas'])
        self._create_metadataset(self._data_locs['antennas_meta'])
        antenna_metadatas = []
        for i, antenna in enumerate(detector):
            antenna_metadata = antenna._metadata
            for j, key in enumerate(data.attrs['keys']):
                data[i, j] = antenna_metadata.pop(self._decode_attr(key))
            antenna_metadatas.append(antenna_metadata)
        self._write_metadata(self._data_locs['antennas_meta'],
                             antenna_metadatas)

    @property
    def has_detector(self):
        """
        Boolean of whether the file has a linked detector.

        """
        return hasattr(self, "_detector")

    def _write_particles(self, event, throw_count=1):
        """
        Write the particle metadata of the particles in `event`.

        Increments the particle counter and writes metadata to the
        'particles_meta' metadata group datasets.

        Parameters
        ----------
        event : Event
            `Event` object whose metadata will be written to the file.
        throw_count : int, optional
            Number of events thrown to reach this event, i.e. value to add to
            the total count of events thrown.

        """
        start_index = self._counters['particles_meta']
        self._counters['particles_meta'] += len(event)
        metadata = self._create_metadataset(self._data_locs['particles_meta'])
        # Reshape metadata datasets to accommodate the event
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['particles_meta'], axis=0)
        float_data.resize(self._counters['particles_meta'], axis=0)

        self._write_indices(self._data_locs['particles_meta'],
                            start_index, len(event))
        self._write_metadata(self._data_locs['particles_meta'],
                             event._metadata, start_index)
        try:
            metadata.attrs['total_thrown'] += throw_count
        except KeyError:
            metadata.attrs['total_thrown'] = throw_count


    @staticmethod
    def _check_trigger(triggered):
        """
        Check the global detector trigger status given a ``bool`` or ``dict``.

        Parameters
        ----------
        triggered : bool or dict
            If ``bool``, boolean value represents the global trigger directly.
            If ``dict``, 'global' key's value represents the global trigger.

        Returns
        -------
        bool
            Global trigger status of the detector.

        Raises
        ------
        ValueError
            If `triggered` is a dictionary and doesn't contain a 'global' key.

        """
        if isinstance(triggered, bool):
            return triggered
        elif isinstance(triggered, dict):
            if "global" not in triggered:
                raise ValueError("Dictionary of triggers must include 'global'")
            return triggered["global"]
        else:
            raise TypeError("Unsupported type for 'triggered' ("+
                            str(type(triggered))+")")

    def _write_trigger(self, triggered, include_antennas=False):
        """
        Write the trigger data of the event.

        Increments the trigger and waveform trigger counters and writes data
        to the 'triggers' and 'waveform_triggers' datasets.

        Parameters
        ----------
        triggered : bool or dict
            If ``bool``, represents the global trigger status. If ``dict``,
            'global' key's value should represent the global trigger status and
            other keys may represent other triggers to be recorded. The other
            keys may be a single boolean for the event, or a list of booleans
            per waveform.
        include_antennas : bool, optional
            Whether to calculate and include individual antenna triggers for
            each antenna in the detector, for each waveform.

        """
        self._counters['triggers'] += 1

        global_trigger = self._check_trigger(triggered)
        if isinstance(triggered, bool):
            extra_triggers = False
        else:
            other_triggered = {key: val for key, val in triggered.items()
                               if key!="global"}
            extra_triggers = len(other_triggered)>0

        # Write global trigger
        trigger_data = self._create_dataset(self._data_locs['triggers'])
        trigger_data.resize(self._counters['triggers'], axis=0)

        trigger_data[self._counters['triggers']-1] = global_trigger
        self._write_indices(self._data_locs['triggers'],
                            self._counters['triggers']-1)

        # Write extra triggers
        if include_antennas or extra_triggers:
            if not self.has_detector:
                raise ValueError("File does not have an associated detector")
            max_waves = max(len(ant.all_waveforms) for ant in self._detector)
            start_index = self._counters['mc_triggers']
            self._counters['mc_triggers'] += max_waves

            extra_data = self._create_dataset(self._data_locs['mc_triggers'])
            extra_data.resize(self._counters['mc_triggers'], axis=0)

            # Add keys that don't exist yet
            extra_keys = []
            if include_antennas:
                extra_keys.extend(["antenna_"+str(i) for i in
                                   range(len(self._detector))])
            if extra_triggers:
                extra_keys.extend(other_triggered.keys())
            for key in extra_keys:
                if self._encode_attr(key) not in list(extra_data.attrs['keys']):
                    extra_data.attrs['keys'] = (list(extra_data.attrs['keys'])
                                                + [self._encode_attr(key)])
                    extra_data.resize(extra_data.shape[1]+1, axis=1)

            # Store individual antenna triggers
            if include_antennas:
                for i, ant in enumerate(self._detector):
                    for k, match in enumerate(extra_data.attrs['keys']):
                        if "antenna_"+str(i)==self._decode_attr(match):
                            for j, wave in enumerate(ant.all_waveforms):
                                extra_data[start_index+j, i] = ant.trigger(wave)

            # Store extra triggers
            if extra_triggers:
                for key, val in other_triggered.items():
                    for k, match in enumerate(extra_data.attrs['keys']):
                        if key==self._decode_attr(match):
                            if isinstance(val, bool):
                                for j in range(max_waves):
                                    extra_data[start_index+j, k] = val
                            else:
                                for j in range(max_waves):
                                    extra_data[start_index+j, k] = val[j]

            self._write_indices(self._data_locs['mc_triggers'],
                                start_index, max_waves)


    def _write_ray_data(self, ray_paths, polarizations):
        """
        Write the ray and polarization metadata of each ray solution.

        Increments the ray counter and writes metadata to the 'rays_meta'
        metadata group datasets.

        Parameters
        ----------
        ray_paths : list of list of RayPath
            `RayPath` objects for each antenna, for each ray solution from the
            event.
        polarizations : list of list of array_like
            Vector polarizations for each antenna, for each ray solution from
            the event.

        Raises
        ------
        ValueError
            If the shapes and sizes of `ray_paths` and `polarizations` don't
            match each other and the linked detector.

        """
        if not self.has_detector:
            raise ValueError("File does not have an associated detector")
        if len(ray_paths)!=len(self._detector):
            raise ValueError("Ray paths length doesn't match detector size"+
                             " ("+str(len(ray_paths))+"!="+
                             str(len(self._detector))+")")
        if len(polarizations)!=len(self._detector):
            raise ValueError("Polarizations length doesn't match detector size"+
                             " ("+str(len(polarizations))+"!="+
                             str(len(self._detector))+")")
        for i, paths in enumerate(ray_paths):
            if len(paths)!=len(polarizations[i]):
                raise ValueError("Polarizations length doesn't match "+
                                 "number of ray paths ("+
                                 str(len(polarizations[i]))+"!="+
                                 str(len(paths))+")")
        max_waves = max(len(paths) for paths in ray_paths)
        start_index = self._counters['rays_meta']
        self._counters['rays_meta'] += max_waves
        metadata = self._create_metadataset(self._data_locs['rays_meta'])
        # Reshape metadata datasets to accommodate the ray data of each solution
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['rays_meta'], axis=0)
        float_data.resize(self._counters['rays_meta'], axis=0)

        for i in range(max_waves):
            ray_metadata = []
            for paths, pols in zip(ray_paths, polarizations):
                if i<len(paths):
                    metadata = paths[i]._metadata
                    metadata.update({
                        "polarization_x": pols[i][0],
                        "polarization_y": pols[i][1],
                        "polarization_z": pols[i][2],
                    })
                    ray_metadata.append(metadata)
                else:
                    ray_metadata.append({})
            self._write_metadata(self._data_locs['rays_meta'], ray_metadata,
                                 start_index+i)

        self._write_indices(self._data_locs['rays_meta'],
                            start_index, max_waves)

    def _get_noise_bases(self, antenna):
        """
        Grab noise-generation basis data from an `Antenna` or `AntennaSystem`.

        Finds the lowest-level antenna of `antenna` and gets the data required
        to reproduce its `noise_master`.

        Parameters
        ----------
        antenna : Antenna or AntennaSystem
            Antenna containing (at some depth) a noise master to be stored.

        Returns
        -------
        freqs, amps, phases : array_like
            Frequencies and corresponding amplitudes and phases of the noise,
            making up the "noise basis".

        """
        while hasattr(antenna, "antenna"):
            antenna = antenna.antenna
        noise = antenna._noise_master
        if noise is None:
            return [], [], []
        else:
            return noise.freqs, noise.amps, noise.phases

    def _write_noise_data(self):
        """
        Write the noise basis data of the linked detector for the event.

        Increments the noise counter and writes data to the 'noise' dataset.

        """
        if not self.has_detector:
            raise ValueError("File does not have an associated detector")
        self._counters['noise'] += 1
        data = self._create_dataset(self._data_locs['noise'])
        data.resize(self._counters['noise'], axis=0)

        self._write_indices(self._data_locs['noise'], self._counters['noise']-1)
        for i, ant in enumerate(self._detector):
            data[self._counters['noise']-1, i] = self._get_noise_bases(ant)


    def _write_waveforms(self):
        """
        Write the waveform data of the linked detector for the event.

        Increments the waveform counter and writes data to the 'waveforms'
        dataset.

        """
        if not self.has_detector:
            raise ValueError("File does not have an associated detector")
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        start_index = self._counters['waveforms']
        self._counters['waveforms'] += max_waves

        data = self._create_dataset(self._data_locs['waveforms'])
        data.resize(self._counters['waveforms'], axis=0)

        for i, ant in enumerate(self._detector):
            for j, wave in enumerate(ant.all_waveforms):
                data[start_index+j, i] = np.array([wave.times, wave.values])

        self._write_indices(self._data_locs['waveforms'],
                            start_index, max_waves)


    def add(self, event, triggered=None, ray_paths=None, polarizations=None,
            events_thrown=1):
        """
        Add the data from an event to the file.

        Increments the global event counter and writes data to the file based
        on the specifications given on initialization.

        Parameters
        ----------
        event : Event
            `Event` object responsible for the signals currently on the
            linked detector.
        triggered : bool or dict, optional
            If ``bool``, represents the global trigger status. If ``dict``,
            'global' key's value should represent the global trigger status and
            other keys may represent other triggers to be recorded. The other
            keys may be a single boolean for the event, or a list of booleans
            per waveform.
        ray_paths : list of list of RayPath, optional
            `RayPath` objects for each antenna, for each ray solution from the
            event.
        polarizations : list of list of array_like, optional
            Vector polarizations for each antenna, for each ray solution from
            the event.
        events_thrown : int, optional
            Number of events thrown to reach this event, i.e. how much to
            increase the total throw count of the file by.

        Raises
        ------
        ValueError
            If not enough keyword arguments have been specified to write the
            necessary data.

        """
        if not self.is_open:
            raise IOError("File is not open")

        if self._write_data['rays'] and (ray_paths is None or
                                         polarizations is None):
            raise ValueError("Ray path and polarization information must be "+
                             "provided if writing ray data")
        if np.any(list(self._trig_only.values())) and triggered is None:
            if triggered is None:
                raise ValueError("Trigger information must be provided if "
                                 "writing only when triggered")

        self._preset_all_indices()

        if (self._write_data['particles'] and
                (not self._trig_only['particles']
                 or self._check_trigger(triggered))):
            self._write_particles(event, events_thrown)

        if (self._write_data['triggers'] and
                (not self._trig_only['triggers']
                 or self._check_trigger(triggered))):
            include_antennas = (self._write_data['antenna_triggers'] and
                                (not self._trig_only['antenna_triggers']
                                 or self._check_trigger(triggered)))
            self._write_trigger(triggered, include_antennas)

        if (self._write_data['rays'] and
                (not self._trig_only['rays']
                 or self._check_trigger(triggered))):
            self._write_ray_data(ray_paths, polarizations)

        if (self._write_data['noise'] and
                (not self._trig_only['noise']
                 or self._check_trigger(triggered))):
            self._write_noise_data()

        if (self._write_data['waveforms'] and
                (not self._trig_only['waveforms']
                 or self._check_trigger(triggered))):
            self._write_waveforms()

        self._counters['indices'] += 1


    def create_analysis_group(self, name, *args, **kwargs):
        """
        Create the given analysis group in the file.

        The name given will be forced into an appropriate analysis location if
        it isn't already a valid location. If the given group already exists,
        will simply return the existing group rather than recreating it.

        Parameters
        ----------
        name : str
            Name of the group to be created. May contain the full path to the
            group location.

        Returns
        -------
        group
            The group object corresponding to the created (or existing) group
            requested.

        """
        if not self.is_open:
            raise IOError("File is not open")
        location = self._analysis_location(name)
        return self._file.create_group(location, *args, **kwargs)

    def create_analysis_dataset(self, name, *args, **kwargs):
        """
        Create the given analysis dataset in the file.

        The name given will be forced into an appropriate analysis location if
        it isn't already a valid location. If the given dataset already exists,
        will simply return the existing dataset rather than recreating it.

        Parameters
        ----------
        name : str
            Name of the dataset to be created. May contain the full path to the
            dataset location.

        Returns
        -------
        dataset
            The dataset object corresponding to the created (or existing)
            dataset requested.

        """
        if not self.is_open:
            raise IOError("File is not open")
        location = self._analysis_location(name)
        return self._file.create_dataset(location, *args, **kwargs)

    def create_analysis_metadataset(self, name, *args, **kwargs):
        """
        Create the given analysis metadata group in the file.

        Creates metadata groups with 'float' and 'str' datasets. The name given
        will be forced into an appropriate analysis location if it isn't
        already a valid location. If the given metadata group already exists,
        will simply return the existing group rather than recreating it.

        Parameters
        ----------
        name : str
            Name of the metadata group to be created. May contain the full path
            to the metadata group location.
        shape : tuple of int, optional
            Shape of the metadata datasets to be created for analysis metadata
            (not including the attribute dimension). Attribute dimension only
            by default.
        maxshape : tuple of int, optional
            Maxshape of the metadata datasets to be created for analysis
            metadata (not including the attribute dimension). No limits by
            default.

        Returns
        -------
        group
            The group object corresponding to the created (or existing)
            metadata group requested (containing 'float' and 'str' datasets).

        """
        if not self.is_open:
            raise IOError("File is not open")
        location = self._analysis_location(name)
        return self._create_metadataset(location, *args, **kwargs)

    def add_analysis_metadata(self, name, metadata, index=None):
        """
        Writes the given `metadata` to the analysis metadata group at `name`.

        Splits information in the `metadata` into string and float values and
        stores them in the appropriate datasets of the metadata group. Creates
        new attribute columns for attributes which have not yet been written.
        The name given will be forced into an appropriate analysis location if
        it isn't already a valid location.

        Parameters
        ----------
        name : str
            Name of the metadata group to be written to. May contain the full
            path to the metadata group location.
        metadata : dict or list of dict
            Dictionaries containing attribute column names as keys and float or
            string data as values.
        index : int, optional
            Additional index for specifying the row where metadata should be
            written, used by some higher-dimension metadata groups.

        Raises
        ------
        ValueError
            If `name` is not an existing metadata group.

        """
        if not self.is_open:
            raise IOError("File is not open")
        location = self._analysis_location(name)
        return self._write_metadata(location, metadata, index=index)

    def add_analysis_indices(self, name, global_index,
                             start_index, length=1):
        """
        Write the given start and length indices to the event indices dataset.

        Parameters
        ----------
        name : str
            Location and name of the dataset or metadata group to be indexed.
            Will be forced into an appropriate analysis location if it isn't
            already a valid location.
        global_index : int
            Global event index (row of the event indices dataset) to write to.
        start_index : int
            Starting index in the dataset(s) containing information on the
            event.
        length : int, optional
            Number of rows in the dataset(s) containing data for the event.

        """
        if not self.is_open:
            raise IOError("File is not open")
        location = self._analysis_location(name)
        return self._write_indices(location, start_index, length=length,
                                   global_index_value=global_index)

    def add_file_metadata(self, metadata):
        """
        Writes the given `metadata` to the general file metadata group.

        Splits information in the `metadata` into string and float values and
        stores them in the appropriate datasets of the file metadata group.

        Parameters
        ----------
        metadata : dict or list of dict
            Dictionary containing attribute column names as keys and float or
            string data as values.

        """
        if not self.is_open:
            raise IOError("File is not open")
        self._write_metadata(self._data_locs['file_meta'], metadata)



class File:
    """
    Class for reading or writing data files.

    Works as a context manager and allows for reading or writing
    simulation/real data or analysis-level data to the given file. Chooses
    the appropriate class for handling the given file type.

    Parameters
    ----------
    filename : str
        File name to open in the given write mode.
    mode : str, optional
        Mode with which to open the file.
    **kwargs
        Keyword arguments passed on to the appropriate file handler.

    Attributes
    ----------
    readers : dict
        Dictionary with file extensions as keys and values with the
        corresponding classes used to handle reading of those file types.
    writers : dict
        Dictionary with file extensions as keys and values with the
        corresponding classes used to handle writing of those file types.

    See Also
    --------
    pyrex.io.HDF5Reader : Class for reading data from an hdf5 file.
    pyrex.io.HDF5Writer : Class for writing data to an hdf5 file.

    """
    readers = {
        "h5": HDF5Reader,
        "hdf5": HDF5Reader,
    }

    writers = {
        "h5": HDF5Writer,
        "hdf5": HDF5Writer,
    }

    def __new__(cls, filename, mode='r', **kwargs):
        if '.' not in filename:
            raise ValueError("Unable to handle filenames without extensions")
        _, suffix = filename.rsplit(".", 1)

        if mode=='r':
            if suffix not in cls.readers:
                raise ValueError("Unable to read filenames with extension '."
                                 +suffix+"'")
            return cls.readers[suffix](filename, **kwargs)

        else:
            if suffix not in cls.writers:
                raise ValueError("Unable to write filenames with extension '."
                                 +suffix+"'")
            return cls.writers[suffix](filename, mode=mode, **kwargs)
