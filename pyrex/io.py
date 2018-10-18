"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

import datetime
import inspect
import logging
import h5py
import numpy as np
from pyrex.__about__ import __version__
from pyrex.internal_functions import get_from_enum
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.particle import Particle, Event

logger = logging.getLogger(__name__)


class BaseReader:
    def __init__(self, filename):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    @property
    def is_open(self):
        raise NotImplementedError


class BaseWriter:
    def __init__(self, filename, **kwargs):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def is_open(self):
        raise NotImplementedError

    def set_detector(self):
        raise NotImplementedError

    @property
    def has_detector(self):
        raise NotImplementedError


class BaseRebuilder:
    def __init__(self, filename):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        self._detector = None
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def is_open(self):
        raise NotImplementedError

    def __iter__(self):
        self._iter_counter = -1
        return self

    def __next__(self):
        self._iter_counter += 1
        if self._iter_counter < len(self):
            return self[self._iter_counter]
        else:
            raise StopIteration

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        event = self.rebuild_event(key)
        self.rebuild_waveforms(key)
        return event, self._detector

    def rebuild_event(self, index):
        raise NotImplementedError

    def rebuild_detector(self):
        raise NotImplementedError

    def rebuild_waveforms(self, index):
        raise NotImplementedError



class HDF5Base:
    def __init__(self, file_version_major, file_version_minor):
        self._file_version_major = file_version_major
        self._file_version_minor = file_version_minor

    def _dataset_locations(self):
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

    def _analysis_location(self, name):
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

    @staticmethod
    def _read_metadata_to_dicts(file, name, index=None):
        # TODO: Make this function work with data or file objects
        str_metadata = file[name]['str']
        float_metadata = file[name]['float']

        if index is None:
            str_table = str_metadata
            float_table = float_metadata
        else:
            str_table = str_metadata[index]
            float_table = float_metadata[index]

        if str_table.ndim!=float_table.ndim:
            raise ValueError("Metadata group '"+name+"' not readable")

        ndim = str_table.ndim
        key_dim = -1
        if ndim==0:
            return {}

        if (str_table.shape[key_dim]!=len(str_metadata.attrs['keys']) or
                float_table.shape[key_dim]!=len(float_metadata.attrs['keys'])):
            raise ValueError("Metadata group '"+name+"' not readable")

        # Recursively pull out list (of lists ...) of dictionaries
        def get_dicts_recursive(dimension, str_data, float_data):
            if dimension==key_dim%ndim:
                meta_dict = {}
                for j, key in enumerate(str_metadata.attrs['keys']):
                    meta_dict[bytes.decode(key)] = str_data[j]
                for j, key in enumerate(float_metadata.attrs['keys']):
                    meta_dict[bytes.decode(key)] = float_data[j]
                return meta_dict
            else:
                if str_data.shape[0]!=float_data.shape[0]:
                    raise ValueError("Metadata group '"+name+"' not readable")
                return [
                    get_dicts_recursive(dimension+1, str_data[i], float_data[i])
                    for i in range(str_data.shape[0])
                ]

        return get_dicts_recursive(0, str_table, float_table)

    @staticmethod
    def _read_datasets_to_dicts(file, name, index=None):
        table_metadata = file[name]
        if index is None:
            table = table_metadata
        else:
            table = table_metadata[index]
        ndim = table.ndim
        key_dim = -1

        # Recursively pull out list (of lists ...) of dictionaries
        def get_dicts_recursive(dimension, data):
            if dimension == key_dim % ndim:
                meta_dict = {}
                for j, key in enumerate(table_metadata.attrs['keys']):
                    meta_dict[bytes.decode(key)] = data[j]
                return meta_dict
            else:
                return [
                    get_dicts_recursive(
                        dimension+1, table[i])
                    for i in range(table.shape[0])
                ]
        return get_dicts_recursive(0,table)

    @staticmethod
    def _get_bool_dict(file, locations):
        return {key: group in file and file[group].size>0
                for key, group in locations.items()}

    @staticmethod
    def _get_keys_dict(file, group):
        if group in file and "keys" in file[group].attrs.keys():
            return {str(key, "utf-8"): i for i, key in
                    enumerate(file[group].attrs["keys"])}
        else:
            return {}

    @staticmethod
    def _convert_ray_value(ray_value):
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
    def __init__(self, hdf5_file, slice_range=10,
                 start_event=None, stop_event=None, step=None):
        self._object = hdf5_file
        HDF5Base.__init__(self, self._object.attrs["version_major"],
                          self._object.attrs["version_minor"])

        self._locations_original = self._dataset_locations()
        self._locations = {}
        for key, value in self._locations_original.items():
            if key.endswith("meta"):
                self._locations[key+"_str"] = value+"/str"
                self._locations[key+"_float"] = value+"/float"
            else:
                self._locations[key] = value

        self._max_antenna = max(
            len(self._object[self._locations["antennas_meta_float"]]),
            len(self._object[self._locations["antennas_meta_str"]])
        )
        self._max_events = hdf5_file[self._locations["indices"]].shape[0]

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
            str(key, "utf-8")
            for key in self._object[self._locations["indices"]].attrs["keys"]
        ]

        self._keys = {}
        for key, value in self._locations.items():
            self._keys[key] = self._get_keys_dict(self._object, value)

    def _get_index_from_list(self, string_to_comp, keys):
        if string_to_comp == self._locations["indices"]:
            raise ValueError("This is the index dataset")
        for i, key in enumerate(keys):
            if string_to_comp in key:
                return i
        return -1

    def __iter__(self):
        return self

    def __next__(self):
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
        # Store event indices first
        slc = slice(self._slice_start_event,
                    self._slice_end_event,
                    self._slice_step)
        self._data['indices'] = self._object[self._locations['indices']]
        for key, val in self._locations.items():
            if key=="indices" or not self._bool_dict[key]:
                continue
            parts = key.split("_")
            if parts[-1]=="float" or parts[-1]=="str":
                parts.pop(-1)
            key_org = "_".join(parts)
            val_org = self._locations_original[key_org]
            index = self._get_index_from_list(val_org, self._index_keys)
            if index>=0:
                self._data[key] = []
                # Could probably optimize this to only read once from each
                # dataset rather than N times
                for start, length in self._data['indices'][slc, index]:
                    self._data[key].append(self._object[val][start:start+length])


    def _confirm_iterating(self):
        if self._slice_start_event == self._slice_end_event:
            raise ValueError("This function is should be called after "+
                             "initializing the iterator object and calling "+
                             "next(<iterator>)")


    def get_waveforms(self, antenna_id=None, waveform_type=None):
        self._confirm_iterating()

        data = self._get_event_data("waveforms")

        if antenna_id is None:
            antenna_id = slice(None)
        elif isinstance(antenna_id, int) and antenna_id > self._max_antenna:
            raise ValueError("Antenna Id provided is greater than the number "+
                             "of antennas in detector")

        wf_index = self._convert_ray_value(waveform_type)
        if wf_index is None:
            return data[:, antenna_id]
        elif isinstance(wf_index, int):
            if wf_index>=data.shape[0]:
                return []
            return data[wf_index, antenna_id]
        else:
            raise ValueError("The parameter 'waveform_type' should either be "+
                             "an integer or a string")


    def _get_event_data(self, group):
        if group.endswith("meta"):
            float_data = (np.array([]) if not self._bool_dict[group+"_float"]
                          else self._data[group+"_float"][self._iter_counter])
            str_data = (np.array([]) if not self._bool_dict[group+"_str"]
                        else self._data[group+"_str"][self._iter_counter])
            return float_data, str_data
        else:
            return (np.array([]) if not self._bool_dict[group]
                    else self._data[group][self._iter_counter])


    def get_particle_info(self, attribute=None):
        custom_values = ["position", "vertex", "direction", "interaction_info"]
        # Possibly add distance and radius to custom_values
        self._confirm_iterating()

        float_data, str_data = self._get_event_data("particles_meta")

        if len(float_data)==0 and len(str_data)==0:
            logger.debug("No particle data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)

        if attribute is None:
            # SPEED CONCERN : This directly accesses the file, try to avoid it
            # as much as you can
            raise NotImplementedError
            return self._read_metadata_to_dicts(
                self._object,
                self._locations_original["particles_meta"],
                iter_counter
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
                    for grp, data in zip(groups, datasets):
                        for key, value in self._keys[grp].items():
                            if "interaction" in key:
                                dic[key] = data[:, value]
                    return dic
                else:
                    raise ValueError("This value is not supported yet")
            else:
                if attribute in self._keys["particles_meta_float"]:
                    index = self._keys["particles_meta_float"][attribute]
                    return float_data[:, index]
                elif attribute in self._keys["particles_meta_str"]:
                    index = self._keys["particles_meta_str"][attribute]
                    return str_data[:, index]
                else:
                    raise ValueError("Unrecognized particle attribute '"+
                                     attribute+"'")

        else:
            raise ValueError("Only string values supported as argument")


    def get_rays_info(self, attribute=None):
        custom_values = ["polarization", "emitted_direction",
                         "received_direction"]
        self._confirm_iterating()

        float_data, str_data = self._get_event_data("rays_meta")

        if len(float_data)==0 and len(str_data)==0:
            logger.debug("No ray data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)

        if attribute is None:
            # Make sure that _read_hdf5_metadata function account for
            # multiple rays
            # SPEED WARNING: Try to avoid this function, this accesses the file
            # directly. Can consider writing another function to return the
            # dictionaries using the stored data
            raise NotImplementedError
            return self._read_metadata_to_dicts(
                self._object,
                self._locations_original["rays_meta"],
                iter_counter
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
                    return str_data[:, :, index]
                else:
                    raise ValueError("Unrecognized particle attribute '"+
                                     attribute+"'")
        else:
            raise ValueError("Only string values supported as argument")

    @property
    def triggered(self):
        self._confirm_iterating()
        triggered = self._get_event_data("triggers")
        if len(triggered)==0:
            return None
        else:
            return triggered[0]

    @property
    def noise_bases(self):
        self._confirm_iterating()
        bases = self._get_event_data("noise")
        if len(bases)==0:
            logger.debug("No noise data was stored for event %s",
                         self._iter_counter * self._slice_step
                         + self._slice_start_event)
            return bases
        else:
            return bases[0]

    def get_triggered_components(self, ray=None):
        """Returns the list of trigger components which were triggered in the
        event """
        self._confirm_iterating()
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
        """Returns the flavor of the initial particle. If not a neutrino, then
        returns a null string"""
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
        """Returns true if the initial particle was anti-neutrino, returns
        false if the initial particle was neutrino, else returns None"""
        if self.is_neutrino:
            return self.get_particle_info("particle_id")[0] < 0

    @property
    def is_neutrino(self):
        return "neutrino" in self.get_particle_info("particle_name")[0]



class HDF5Reader(BaseReader,HDF5Base):
    def __init__(self, filename, slice_range=10):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
            self._slice_range = slice_range
        else:
            raise ValueError(filename+" is not in the HDF5 format")
        self._is_open = False

    def __getitem__(self, key):
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
            if key in self._locations_original:
                return self._file[self._locations_original[key]]
            elif key in self._locations:
                return self._file[self._locations[key]]
            else:
                return self._file[key]
        else:
            raise ValueError("Invalid key '"+str(key)+"'")

    def __len__(self):
        return self._num_events

    def __iter__(self):
        return EventIterator(hdf5_file=self._file,
                             slice_range=self._slice_range)

    def open(self):
        self._file = h5py.File(self.filename, mode='r')
        self._is_open = True
        HDF5Base.__init__(self, self._file.attrs["version_major"],
                          self._file.attrs["version_minor"] )
        self._locations_original = self._dataset_locations()
        self._locations = {}
        for key, value in self._locations_original.items():
            if key.endswith("meta"):
                self._locations[key+"_str"] = value+"/str"
                self._locations[key+"_float"] = value+"/float"
            else:
                self._locations[key] = value

        self._num_ant = 0
        for key in ["antennas_meta_float", "antennas_meta_str"]:
            loc = self._locations[key]
            if loc in self._file:
                self._num_ant = max(self._num_ant, len(self._file[loc]))
        self._iter_counter = None
        # assumming that the data indices will have all the list of all events
        self._num_events = 0
        if self._locations["indices"] in self._file:
            self._num_events = len(self._file[self._locations["indices"]])

        self._bool_dict = self._get_bool_dict(self._file, self._locations)


        def get_keys_dict(self, group_addr, dataset):
            dic = {}
            count = 0
            for key in self._file[group_addr][dataset].attrs["keys"]:
                key = str(key, "utf-8")
                dic[key] = count
                count += 1
            return dic

        self._event_indices_key = self._get_keys_dict(self._file,
                                                      self._locations["indices"])

        self._event_data = None
        if self._bool_dict["waveforms"]:
            self._event_data = self._file[self._locations["waveforms"]]

    def close(self):
        self._file.close()

    @property
    def is_open(self):
        return self._is_open

    def _get_table_slice(self, group, event_indices):
        start = self._file[self._locations["indices"]][event_indices,
                                                       self._event_indices_key[group],
                                                       0]
        length = self._file[self._locations["indices"]][event_indices,
                                                        self._event_indices_key[group],
                                                        1]
        return slice(start, start+length)


    def _get_waveform_of_one_kind(self, wf_type):
        logger.warn("Getting all waveforms of a single type requires "+
                    "iteration over the entire dataset. Consider iterating "+
                    "the file manually instead.")
        return np.asarray(
            [event.get_waveforms(waveform_type=wf_type) for event in self]
        )

    def get_waveforms(self, event_id=None, antenna_id=None, waveform_type=None):
        if self._event_data is None:
            raise ValueError("The waveforms were not saved for this run. "+
                             "Please check the run configuration")

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
        elif isinstance(antenna_id, int) and antenna_id > self._num_ant:
            raise ValueError("Antenna Id provided is greater than the number "+
                             "of antennas in detector")

        wf_index = self._convert_ray_value(waveform_type)
        if wf_index is None:
            return self._event_data[event_id, antenna_id]
        elif isinstance(wf_index, int):
            start = event_id.start
            stop = event_id.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self._event_data.shape[0]
            if wf_index>=stop-start:
                raise ValueError("Invalid waveform index ("+str(wf_index)+")")
            return self._event_data[start+wf_index, antenna_id]
        else:
            raise ValueError("The parameter 'waveform_type' should either be "+
                             "an integer or a string")

    @classmethod
    def _recursive_combine_dicts(cls, dicts1, dicts2):
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
        ant_dict_data = self._read_datasets_to_dicts(
            self._file, "/data/antennas"
        )
        ant_dict_MC = self._read_metadata_to_dicts(
            self._file, "/monte_carlo_data/antennas"
        )
        return self._recursive_combine_dicts(ant_dict_data, ant_dict_MC)

    @property
    def file_metadata_info(self):
        return self._read_metadata_to_dicts(self._file, "file_metadata")



class HDF5Writer(BaseWriter, HDF5Base):
    def __init__(self, filename, mode='x', write_particles=True,
                 write_triggers=True, write_antenna_triggers=False,
                 write_rays=True, write_noise=False, write_waveforms=False,
                 require_trigger=True):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            self.filename = filename+".h5"
        if mode not in ['w', 'x', 'a', 'r+']:
            raise ValueError("Uncrecognized file mode '"+str(mode)+"'")
        self._mode = mode
        self._is_open = False

        # Set file version
        HDF5Base.__init__(self, 1, 0)
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

    def open(self):
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
            "file_version": "1.0",
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
        self._file.close()
        self._is_open = False

    @property
    def is_open(self):
        return self._is_open


    def _create_dataset(self, name):
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
            data.attrs['keys'] = [
                b"position_x", b"position_y", b"position_z",
                b"z_axis_x", b"z_axis_y", b"z_axis_z",
                b"x_axis_x", b"x_axis_y", b"x_axis_z"
            ]
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
            data.attrs['keys'] = [b"frequency", b"amplitude", b"phase"]
            data.resize(len(data.attrs['keys']), axis=2)

        else:
            raise ValueError("Unrecognized dataset name '"+name+"'")

        return data


    def _create_metadataset(self, name, shape=None, maxshape=None):
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
                        if bytes.decode(match)==key:
                            break
                    else:
                        # If no key matched, add key and resize dataset
                        j = len(str_data.attrs['keys'])
                        str_data.attrs['keys'] = np.append(
                            str_data.attrs['keys'], str.encode(key)
                        )
                        str_data.resize(j+1, axis=data_axis)
                    write_value(val, str_data, j, i, index)
                elif val_type=="float":
                    for j, match in enumerate(float_data.attrs['keys']):
                        if bytes.decode(match)==key:
                            break
                    else:
                        # If no key matched, add key and resize dataset
                        j = len(float_data.attrs['keys'])
                        float_data.attrs['keys'] = np.append(
                            float_data.attrs['keys'], str.encode(key)
                        )
                        float_data.resize(j+1, axis=data_axis)
                    write_value(val, float_data, j, i, index)
                else:
                    raise ValueError("Unrecognized val_type")


    def _write_indices(self, full_name, start_index, length=1,
                       global_index_value=None):
        if global_index_value is None:
            global_index_value = self._counters['indices']
        indices = self._file[self._data_locs['indices']]
        # Don't write indices if the matching dataset doesn't exist
        if full_name not in self._file:
            return
        encoded_name = str.encode(full_name)
        # Add column for full_name if it doesn't exist
        if (len(indices.attrs['keys'])==0 or
                encoded_name not in indices.attrs['keys']):
            indices.attrs['keys'] = np.append(indices.attrs['keys'],
                                              str.encode(full_name))
            indices.resize(indices.shape[1]+1, axis=1)
        for i, key in enumerate(indices.attrs['keys']):
            if key==encoded_name:
                if indices.shape[0]<=global_index_value:
                    indices.resize(global_index_value+1, axis=0)
                indices[global_index_value, i] = (start_index, length)
                return
        raise ValueError("Unrecognized table name '"+full_name+"'")


    def _preset_all_indices(self):
        for key, count in self._counters.items():
            if key=="indices":
                continue
            self._write_indices(self._data_locs[key], count, 0)


    def set_detector(self, detector):
        self._detector = detector
        data = self._create_dataset(self._data_locs['antennas'])
        self._create_metadataset(self._data_locs['antennas_meta'])
        antenna_metadatas = []
        for i, antenna in enumerate(detector):
            antenna_metadata = antenna._metadata
            for j, key in enumerate(data.attrs['keys']):
                data[i, j] = antenna_metadata.pop(bytes.decode(key))
            antenna_metadatas.append(antenna_metadata)
        self._write_metadata(self._data_locs['antennas_meta'],
                             antenna_metadatas)

    @property
    def has_detector(self):
        return hasattr(self, "_detector")

    def _write_particles(self, event):
        start_index = self._counters['particles_meta']
        self._counters['particles_meta'] += len(event)
        metadata = self._create_metadataset(self._data_locs['particles_meta'])
        # Reshape metadata datasets to accomodate the event
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['particles_meta'], axis=0)
        float_data.resize(self._counters['particles_meta'], axis=0)

        self._write_indices(self._data_locs['particles_meta'],
                            start_index, len(event))
        self._write_metadata(self._data_locs['particles_meta'],
                             event._metadata, start_index)


    @staticmethod
    def _check_trigger(triggered):
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
                if str.encode(key) not in list(extra_data.attrs['keys']):
                    extra_data.attrs['keys'] = np.append(
                        extra_data.attrs['keys'], str.encode(key)
                    )
                    extra_data.resize(extra_data.shape[1]+1, axis=1)

            # Store individual antenna triggers
            if include_antennas:
                for i, ant in enumerate(self._detector):
                    for k, match in enumerate(extra_data.attrs['keys']):
                        if "antenna_"+str(i)==bytes.decode(match):
                            for j, wave in enumerate(ant.all_waveforms):
                                extra_data[start_index+j, i] = ant.trigger(wave)

            # Store extra triggers
            if extra_triggers:
                for key, val in other_triggered.items():
                    for k, match in enumerate(extra_data.attrs['keys']):
                        if key==bytes.decode(match):
                            if isinstance(val, bool):
                                for j in range(max_waves):
                                    extra_data[start_index+j, k] = val
                            else:
                                for j in range(max_waves):
                                    extra_data[start_index+j, k] = val[j]

            self._write_indices(self._data_locs['mc_triggers'],
                                start_index, max_waves)


    def _write_ray_data(self, ray_paths, polarizations):
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
        # Reshape metadata datasets to accomodate the ray data of each solution
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
        while hasattr(antenna, "antenna"):
            antenna = antenna.antenna
        noise = antenna._noise_master
        if noise is None:
            return [], [], []
        else:
            return noise.freqs, noise.amps, noise.phases

    def _write_noise_data(self):
        self._counters['noise'] += 1
        data = self._create_dataset(self._data_locs['noise'])
        data.resize(self._counters['noise'], axis=0)

        self._write_indices(self._data_locs['noise'], self._counters['noise']-1)
        for i, ant in enumerate(self._detector):
            data[self._counters['noise']-1, i] = self._get_noise_bases(ant)


    def _write_waveforms(self):
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


    def add(self, event, triggered=None, ray_paths=None, polarizations=None):
        if self._write_data['rays'] and (ray_paths is None or
                                         polarizations is None):
            raise ValueError("Ray path and polarization information must be "+
                             "provided if writing ray data")
        if np.any(self._trig_only.items()) and triggered is None:
            if triggered is None:
                raise ValueError("Trigger information must be provided if "
                                 "writing only when triggered")

        self._preset_all_indices()

        if (self._write_data['particles'] and
                (not self._trig_only['particles']
                 or self._check_trigger(triggered))):
            self._write_particles(event)

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
        location = self._analysis_location(name)
        return self._file.create_group(location, *args, **kwargs)

    def create_analysis_dataset(self, name, *args, **kwargs):
        location = self._analysis_location(name)
        return self._file.create_dataset(location, *args, **kwargs)

    def create_analysis_metadataset(self, name, *args, **kwargs):
        location = self._analysis_location(name)
        return self._create_metadataset(location, *args, **kwargs)

    def add_analysis_metadata(self, name, metadata, index=None):
        location = self._analysis_location(name)
        return self._write_metadata(location, metadata, index=index)

    def add_analysis_indices(self, name, global_index,
                             start_index, length=1):
        location = self._analysis_location(name)
        return self._write_indices(location, start_index, length=length,
                                   global_index_value=global_index)

    def add_file_metadata(self, metadata):
        self._write_metadata(self._data_locs['file_meta'], metadata)


class AntennaProxy(Antenna):
    def __init__(self, metadata_dict):
        position = metadata_dict['position']
        super().__init__(position=position)
        for key, val in metadata_dict.items():
            self.__dict__[key] = val


class HDF5Rebuilder(BaseRebuilder):
    def __init__(self, filename, use_detector_proxy=False):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            raise ValueError(filename+" is not an hdf5 file")
        self.use_proxy = use_detector_proxy

    def open(self):
        self._file = h5py.File(self.filename, mode='r')
        if self.use_proxy:
            self._detector = self.build_proxy_detector()
        else:
            self._detector = self.rebuild_detector()

    def close(self):
        self._file.close()

    def __len__(self):
        return (self._file['events'].shape[0] +
                self._file['complex_events'].shape[0])

    def _read_metadata(self, group, index=None):
        return _read_hdf5_metadata_to_dicts(self._file, group, index)

    def rebuild_event(self, index):
        event_metadata = self._read_metadata('events', index)
        event_roots = []
        for particle_metadata in event_metadata:
            required_keys = [
                'particle_id', 'vertex_x', 'vertex_y', 'vertex_z',
                'direction_x', 'direction_y', 'direction_z', 'energy'
            ]
            for key in required_keys:
                if key not in particle_metadata:
                    raise ValueError(
                        "Event metadata does not have a value for "+key)
            particle_id = particle_metadata['particle_id']
            vertex = (particle_metadata['vertex_x'],
                      particle_metadata['vertex_y'],
                      particle_metadata['vertex_z'])
            direction = (particle_metadata['direction_x'],
                         particle_metadata['direction_y'],
                         particle_metadata['direction_z'])
            energy = particle_metadata['energy']
            p = Particle(particle_id=particle_id, vertex=vertex,
                         direction=direction, energy=energy)
            event_roots.append(p)
        return Event(event_roots)

    def rebuild_detector(self):
        detector_metadata = self._read_metadata('antennas')

    def build_proxy_detector(self):
        proxy_detector = []
        detector_metadata = self._read_metadata('antennas')
        for antenna_metadata in detector_metadata:
            position = (antenna_metadata.pop('position_x'),
                        antenna_metadata.pop('position_y'),
                        antenna_metadata.pop('position_z'))
            antenna_metadata['position'] = position
            proxy_detector.append(AntennaProxy(antenna_metadata))
        return proxy_detector

    def rebuild_waveforms(self, index, detector=None):
        if detector is None:
            detector = self._detector
        for ant in detector:
            ant.clear()
        # waveform_metadata = self._read_metadata('waveforms', index)
        data = self._file['events'][index]
        if data.shape[0] != len(detector):
            raise ValueError("Invalid number of antennas in given detector")

        for i, ant in enumerate(detector):
            for waveform_data in data[i]:
                signal = Signal(waveform_data[0], waveform_data[1])
                ant._all_waves.append(signal)
