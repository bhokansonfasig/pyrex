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
        if self._iter_counter<len(self):
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
        locations = {}
        locations['file_meta'] = "/file_metadata"
        locations['indices'] = "/event_indices"
        locations['waveforms'] = "/data/waveforms"
        locations['triggers'] = "/data/triggers"
        locations['antennas'] = "/data/antennas"
        locations['particles_meta'] = "/monte_carlo_data/particles"
        locations['antennas_meta'] = "/monte_carlo_data/antennas"
        locations['rays_meta'] = "/monte_carlo_data/rays"
        locations['wf_triggers'] = "/monte_carlo_data/triggers"
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


class HDF5Reader(BaseReader):
    def __init__(self, filename):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
            self._file = h5py.File(self.filename, mode='r')
        else:
            raise RuntimeError('Invalid File Format')
    
    def __getitem__(self,given):
        if isinstance(given, slice):
            # do your handling for a slice object:
            #print("slice", given.start, given.stop, given.step)
            return self._file['events'][given]
        elif isinstance(given, tuple):
            return self._file['events'][given]
        elif given == "metadata":
            # Do your handling for a plain index
            print("plain", given)

    def open(self):
        self._file = h5py.File(self.filename, mode='r')

    def close(self):
        self._file.close()

    def get_wf(self,event_id = -1,antenna_id = -1):
        if event_id < 0 or antenna_id < 0:
            raise RuntimeError(
                "Usage: <HDF5Reader>.get_wf(event_id, antenna_id)")
        return np.asarray(self._file['events'][event_id,antenna_id,:,:])
    
    def get_all_wf(self):
        return np.asarray(self._file['events'])
    
    def get_all_wf_from_ant(self,antenna_id = -1, waveform_type=None):
        if antenna_id < 0:
            raise RuntimeError(
                "Usage: <HDF5Reader>.get_all_wf_from_ant(antennaId)")
        return np.asarray(self._file['events'][:,antenna_id,:,:])

#Comibine these two and ask for an index from the user

    def get_all_wf_type(self,wf_type=""):
        print(wf_type.lower())
        if wf_type == "":
            raise RuntimeError(
                "Usage: <HDF5Reader>.get_all_wf_type('direct'/'reflected')")

        elif wf_type.lower() == "direct":
            return self._file[:,:,0,:]
        elif wf_type.lower() == "reflected":
            return self._file[:,:,1,:]
        else:
            raise RuntimeError(
                "Usage: <HDF5Reader>.get_all_wf_type('direct'/'reflected')")



class HDF5Writer(BaseWriter, HDF5Base):
    def __init__(self, filename, write_particles=True, write_waveforms=False,
                 write_triggers=True, write_antenna_triggers=False,
                 write_rays=True, write_noise=False, require_trigger=True):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            self.filename = filename+".h5"
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
        self._file = h5py.File(self.filename, mode='w')
        self._is_open = True
        self._create_dataset(self._data_locs['indices'])
        self._file.attrs['version_major'] = self._file_version_major
        self._file.attrs['version_minor'] = self._file_version_minor

        # Set event number counters
        self._counters = {
            "events": -1,
            "waveforms": -1,
            "triggers": -1,
            "waveform_triggers": -1,
            "particles": -1,
            "rays": -1,
            "noise": -1,
        }

        # Write some generic metadata about the file production
        self._create_metadataset(self._data_locs['file_meta'])
        major, minor, patch = __version__.split('.')
        now = datetime.datetime.now()
        i = 0
        stack = inspect.stack()
        for i, frame in enumerate(stack):
            if frame.function=="open":
                break
        if i<len(stack) and stack[i+1].function=="__enter__":
            if i+1<len(stack):
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
                dtype=np.int_, maxshape=(None, None, 2),
                fillvalue=-1
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
                dtype=np.int_, maxshape=(len(self._detector), None)
            )
            data.dims[0].label = "antennas"
            data.dims[1].label = "attributes"
            data.attrs['keys'] = [
                b"position_x", b"position_y", b"position_z",
                b"z_axis_x", b"z_axis_y", b"z_axis_z",
                b"x_axis_x", b"x_axis_y", b"x_axis_z"
            ]
            data.resize(len(data.attrs['keys']), axis=1)

        elif name==self._data_locs['wf_triggers']:
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
                    if val_length==0:
                        continue
                    if isinstance(val[0], str):
                        val_type = "string"
                    elif np.isscalar(val[0]):
                        val_type = "float"
                    else:
                        raise ValueError("'"+key+"' value ("+str(val[0])+
                                         ") must be string or scalar")
                    if val_length==-1:
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
            global_index_value = self._counters['events']
        indices = self._file[self._data_locs['indices']]
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
        self._counters['events'] += 1
        start_index = self._counters['particles'] + 1
        self._counters['particles'] += len(event)
        metadata = self._create_metadataset(self._data_locs['particles_meta'])
        # Reshape metadata datasets to accomodate the event
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['particles']+1, axis=0)
        float_data.resize(self._counters['particles']+1, axis=0)

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
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        self._counters['triggers'] += 1
        start_index = self._counters['waveform_triggers'] + 1
        self._counters['waveform_triggers'] += max_waves

        global_trigger = self._check_trigger(triggered)
        if isinstance(triggered, bool):
            extra_triggers = False
        else:
            other_triggered = {key: val for key, val in triggered.items()
                               if key!="global"}
            extra_triggers = len(other_triggered)>0

        # Write global trigger
        trigger_data = self._create_dataset(self._data_locs['triggers'])
        trigger_data.resize(self._counters['triggers']+1, axis=0)

        trigger_data[self._counters['triggers']] = global_trigger
        self._write_indices(self._data_locs['triggers'],
                            self._counters['triggers'])

        # Write extra triggers
        if include_antennas or extra_triggers:
            extra_data = self._create_dataset(self._data_locs['wf_triggers'])
            extra_data.resize(self._counters['waveform_triggers']+1, axis=0)

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

            self._write_indices(self._data_locs['wf_triggers'],
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
        start_index = self._counters['rays'] + 1
        self._counters['rays'] += max_waves
        metadata = self._create_metadataset(self._data_locs['rays_meta'])
        # Reshape metadata datasets to accomodate the ray data of each solution
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['rays']+1, axis=0)
        float_data.resize(self._counters['rays']+1, axis=0)

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
        data.resize(self._counters['noise']+1, axis=0)

        self._write_indices(self._data_locs['noise'], self._counters['noise'])
        for i, ant in enumerate(self._detector):
            data[self._counters['noise'], i] = self._get_noise_bases(ant)


    def _write_waveforms(self):
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        start_index = self._counters['waveforms'] + 1
        self._counters['waveforms'] += max_waves

        data = self._create_dataset(self._data_locs['waveforms'])
        data.resize(self._counters['waveforms']+1, axis=0)

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
                    raise ValueError("Event metadata does not have a value for "+key)
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
        if data.shape[0]!=len(detector):
            raise ValueError("Invalid number of antennas in given detector")

        for i, ant in enumerate(detector):
            for waveform_data in data[i]:
                signal = Signal(waveform_data[0], waveform_data[1])
                ant._all_waves.append(signal)
