"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

from enum import Enum
import datetime
import inspect
import h5py
import numpy as np
from pyrex.__about__ import __version__
from pyrex.internal_functions import get_from_enum
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.particle import Particle, Event


class Verbosity(Enum):
    event_data_triggered_only = 0
    basic_triggered_only = 0
    minimum = 0
    event_data = 1
    basic = 1
    ray_data_triggered_only = 2
    debugging_triggered_only = 2
    ray_data = 3
    debugging = 3
    simulation_data_triggered_only = 4
    reproducible_triggered_only = 4
    simulation_data = 5
    reproducible = 5
    reconstruction_data_triggered_only = 6
    waveforms_triggered_only = 6
    reconstruction_data = 7
    waveforms = 7
    all_data_triggered_only = 8
    comlete_triggered_only = 8
    all_data = 9
    complete = 9
    maximum = 9

    default = 2


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
    def __init__(self, filename, verbosity=Verbosity.default):
        pass

    def _set_verbosity_level(self, verbosity, accepted_verbosities):
        if verbosity==-1:
            verbosity = "maximum"
        self.verbosity = get_from_enum(verbosity, Verbosity)
        if self.verbosity not in accepted_verbosities:
            raise ValueError("Unable to write file with verbosity level '"+
                             str(verbosity)+"'")

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



def _read_hdf5_metadata_to_dicts(file, group, index=None):
    str_keys = file['metadata'][group]['str_keys']
    float_keys = file['metadata'][group]['float_keys']
    if index is None:
        str_metadata = file['metadata'][group]['str']
        float_metadata = file['metadata'][group]['float']
    else:
        str_metadata = file['metadata'][group]['str'][index]
        float_metadata = file['metadata'][group]['float'][index]

    if (str_metadata.shape[0]!=float_metadata.shape[0] or
            str_metadata.shape[1]!=str_keys.size or
            float_metadata.shape[1]!=float_keys.size):
        raise ValueError("Metadata group '"+group+"' not readable")

    dicts = []
    for i in range(str_metadata.shape[0]):
        meta_dict = {}
        for j, key in enumerate(str_keys):
            meta_dict[key] = str_metadata[i,j]
        for j, key in enumerate(float_keys):
            meta_dict[key] = float_metadata[i,j]
        dicts.append(meta_dict)
    return dicts


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



class HDF5Writer(BaseWriter):
    def __init__(self, filename, verbosity=Verbosity.default):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            self.filename = filename+".h5"
        accepted_verbosities = [
            Verbosity.event_data_triggered_only,
            Verbosity.event_data,
            Verbosity.ray_data_triggered_only,
            Verbosity.ray_data,
            Verbosity.simulation_data_triggered_only,
            Verbosity.simulation_data,
            Verbosity.reconstruction_data_triggered_only,
            Verbosity.reconstruction_data,
            Verbosity.all_data_triggered_only,
            Verbosity.all_data
        ]
        self._is_open = False
        self._set_verbosity_level(verbosity, accepted_verbosities)
        # Set classifications of verbosity levels
        self._trig_only_verbosities = accepted_verbosities[::2]
        self._event_verbosities = accepted_verbosities
        self._ray_verbosities = accepted_verbosities[2:]
        self._noise_verbosities = (accepted_verbosities[4:6]+
                                   accepted_verbosities[8:])
        self._wave_verbosities = accepted_verbosities[6:]

    def open(self):
        self._file = h5py.File(self.filename, mode='w')
        self._is_open = True
        # Create basic file format
        self._file.create_group("data")
        self._file.create_group("monte_carlo_data")
        self._create_dataset("event_indices")

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
        self._create_metadataset("file_metadata")
        major, minor, patch = __version__.split('.')
        now = datetime.datetime.now()
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
            "file_version_major": 1,
            "file_version_minor": 0,
            "verbosity_name": self.verbosity.name,
            "verbosity_level": self.verbosity.value,
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
        self._write_metadata("file_metadata", metadata)

    def close(self):
        self._file.close()
        self._is_open = False

    @property
    def is_open(self):
        return self._is_open


    def _create_dataset(self, full_name):
        parts = full_name.split("/")
        if len(parts)==1:
            group = "/"
            name = parts[0]
            base = self._file
        elif len(parts)==2:
            group = parts[0]
            name = parts[1]
            base = self._file[group]
        else:
            raise ValueError("Deeply nested datasets not supported")

        # Don't recreate datasets
        if name in base:
            return base[name]

        if group=="/":
            if name=="event_indices":
                data = base.create_dataset(
                    name=name, shape=(0, 0, 2),
                    dtype=np.int_, maxshape=(None, None, 2),
                    fillvalue=-1
                )
                data.dims[0].label = "events"
                data.dims[1].label = "tables"
                data.dims[2].label = "indices"
                data.attrs['keys'] = []
                # data.dims.create_scale(data.attrs['keys'], 'keys')
                # data.dims[1].attach_scale(data.attrs['keys'])
                return

            else:
                raise ValueError("Unrecognized dataset name '"+full_name+"'")

        elif group=="data":
            if name=="waveforms":
                # Dimension lengths:
                #   0 - Total number of waveforms across all events
                #   1 - Number of antennas
                #   2 - Number of value types (2: times & values)
                data = base.create_dataset(
                    name=name, shape=(0, len(self._detector), 2),
                    dtype=h5py.special_dtype(vlen=np.float_),
                    maxshape=(None, len(self._detector), 2)
                )
                data.dims[0].label = "events"
                data.dims[1].label = "antennas"
                data.dims[2].label = "waveforms"

            elif name=="triggers":
                data = base.create_dataset(
                    name=name, shape=(0,),
                    dtype=np.bool_, maxshape=(None,)
                )
                data.dims[0].label = "events"

            elif name=="antennas":
                data = base.create_dataset(
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
                # data.dims.create_scale(data.attrs['keys'], 'keys')
                # data.dims[1].attach_scale(data.attrs['keys'])

            else:
                raise ValueError("Unrecognized dataset name '"+full_name+"'")

        elif group=="monte_carlo_data":
            if name=="noise":
                data = base.create_dataset(
                    name=name, shape=(0, len(self._detector), 3),
                    dtype=h5py.special_dtype(vlen=np.float_),
                    maxshape=(None, len(self._detector), 3)
                )
                data.dims[0].label = "events"
                data.dims[1].label = "antennas"
                data.dims[2].label = "attributes"
                data.attrs['keys'] = [b"frequency", b"amplitude", b"phase"]
                data.resize(len(data.attrs['keys']), axis=2)
                # data.dims.create_scale(data.attrs['keys'], 'keys')
                # data.dims[2].attach_scale(data.attrs['keys'])

            elif name=="triggers":
                data = base.create_dataset(
                    name=name, shape=(0, len(self._detector)),
                    dtype=np.bool_, maxshape=(None, None)
                )
                data.dims[0].label = "events"
                data.dims[1].label = "types"
                data.attrs['keys'] = [str.encode("antenna_"+str(i)) for i in
                                      range(len(self._detector))]
                data.resize(len(data.attrs['keys']), axis=1)
                # data.dims.create_scale(data.attrs['keys'], 'keys')
                # data.dims[1].attach_scale(data.attrs['keys'])

            else:
                raise ValueError("Unrecognized dataset name '"+full_name+"'")

        else:
            raise ValueError("Unrecognized group name '"+group+"'")

        # Add matching column to event_indices
        indices = self._file['event_indices']
        indices.attrs['keys'] = np.append(indices.attrs['keys'], str.encode(full_name))
        indices.resize(indices.shape[1]+1, axis=1)

        return data


    def _create_metadataset(self, full_name):
        parts = full_name.split("/")
        if len(parts)==1:
            group = "/"
            name = parts[0]
            base = self._file
        elif len(parts)==2:
            group = parts[0]
            name = parts[1]
            base = self._file[group]
        else:
            raise ValueError("Deeply nested datasets not supported")

        # Don't recreate meta datasets
        if name in base:
            return base[name]

        if group=="/":
            if name=="file_metadata":
                shape = (0,)
                maxshape = (None,)
                key_dim = 0
                def apply_labels(data):
                    data.dims[0].label = "attributes"

            else:
                raise ValueError("Unrecognized metadataset name '"+full_name+"'")

        elif group=="monte_carlo_data":
            if name=="particles":
                shape = (0, 0)
                maxshape = (None, None)
                key_dim = 1
                def apply_labels(data):
                    data.dims[0].label = "particles"
                    data.dims[1].label = "attributes"

            elif name=="antennas":
                shape = (len(self._detector), 0)
                maxshape = (len(self._detector), None)
                key_dim = 1
                def apply_labels(data):
                    data.dims[0].label = "antennas"
                    data.dims[1].label = "attributes"

            elif name=="rays":
                shape = (0, len(self._detector), 0)
                maxshape = (None, len(self._detector), None)
                key_dim = 2
                def apply_labels(data):
                    data.dims[0].label = "events"
                    data.dims[1].label = "antennas"
                    data.dims[2].label = "attributes"

            else:
                raise ValueError("Unrecognized metadataset name '"+full_name+"'")

        else:
            raise ValueError("Unrecognized group name '"+group+"'")

        # Create group with the given name, create string and float datasets,
        # link attribute names, and label the dimensions
        named_group = base.create_group(name)
        str_data = named_group.create_dataset(
            name="str", shape=shape,
            dtype=h5py.special_dtype(vlen=str), maxshape=maxshape
        )
        # str_keys = named_group.create_dataset(
        #     name="str_keys", shape=(0,),
        #     dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        # )
        # str_data.dims.create_scale(str_keys, 'attribute_names')
        # str_data.dims[key_dim].attach_scale(str_keys)
        str_data.attrs['keys'] = []
        # str_data.dims.create_scale(str_data.attrs['keys'], 'keys')
        # str_data.dims[key_dim].attach_scale(str_data.attrs['keys'])
        apply_labels(str_data)

        float_data = named_group.create_dataset(
            name="float", shape=shape,
            dtype=np.float_, maxshape=maxshape
        )
        # float_keys = named_group.create_dataset(
        #     name="float_keys", shape=(0,),
        #     dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        # )
        # float_data.dims.create_scale(float_keys, 'attribute_names')
        # float_data.dims[key_dim].attach_scale(float_keys)
        float_data.attrs['keys'] = []
        # float_data.dims.create_scale(float_data.attrs['keys'], 'keys')
        # float_data.dims[key_dim].attach_scale(float_data.attrs['keys'])
        apply_labels(float_data)

        # Add matching column to event_indices
        indices = self._file['event_indices']
        indices.attrs['keys'] = np.append(indices.attrs['keys'], str.encode(full_name))
        indices.resize(indices.shape[1]+1, axis=1)

        return named_group


    def _write_metadata(self, full_name, metadata, index=None):
        parts = full_name.split("/")
        if len(parts)==1:
            group = "/"
            name = parts[0]
            base = self._file
        elif len(parts)==2:
            group = parts[0]
            name = parts[1]
            base = self._file[group]
        else:
            raise ValueError("Deeply nested datasets not supported")

        if name not in base:
            raise ValueError("Metadataset '"+full_name+"' does not exist")
        str_data = base[name]['str']
        float_data = base[name]['float']

        if group=="/":
            if name=="file_metadata":
                data_axis = 0
                def write_value(value, dataset, *indices):
                    dataset[indices[0]] = value

            else:
                raise ValueError("Unrecognized metadataset name '"+full_name+"'")

        elif group=="monte_carlo_data":
            if name=="particles":
                data_axis = 1
                def write_value(value, dataset, *indices):
                    dataset[indices[1]+indices[2], indices[0]] = value

            elif name=="antennas":
                data_axis = 1
                def write_value(value, dataset, *indices):
                    dataset[indices[1], indices[0]] = value

            elif name=="rays":
                data_axis = 2
                def write_value(value, dataset, *indices):
                    dataset[indices[2], indices[1], indices[0]] = value

            else:
                raise ValueError("Unrecognized metadataset name '"+full_name+"'")

        else:
            raise ValueError("Unrecognized group name '"+group+"'")

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

                # TODO: else k += 1 and swap k for j
                if val_type=="string":
                    j = -1
                    for k, match in enumerate(str_data.attrs['keys']):
                        if match==key:
                            j = k
                            break
                    if j==-1:
                        j = len(str_data.attrs['keys'])
                        str_data.attrs['keys'] = np.append(str_data.attrs['keys'], str.encode(key))
                        str_data.resize(j+1, axis=data_axis)
                    write_value(val, str_data, j, i, index)
                elif val_type=="float":
                    j = -1
                    for k, match in enumerate(float_data.attrs['keys']):
                        if match==key:
                            j = k
                            break
                    if j==-1:
                        j = len(float_data.attrs['keys'])
                        float_data.attrs['keys'] = np.append(float_data.attrs['keys'], str.encode(key))
                        float_data.resize(j+1, axis=data_axis)
                    write_value(val, float_data, j, i, index)
                else:
                    raise ValueError("Unrecognized val_type")


    def _write_indices(self, full_name, start_index, length=1):
        indices = self._file['event_indices']
        for i, key in enumerate(indices.attrs['keys']):
            if bytes.decode(key)==full_name:
                if indices.shape[0]<=self._counters['events']:
                    indices.resize(self._counters['events']+1, axis=0)
                indices[self._counters['events'], i] = (start_index, length)
                return
        raise ValueError("Unrecognized table name '"+full_name+"'")


    def set_detector(self, detector):
        self._detector = detector
        data = self._create_dataset("data/antennas")
        self._create_metadataset("monte_carlo_data/antennas")
        antenna_metadatas = []
        for i, antenna in enumerate(detector):
            antenna_metadata = antenna._metadata
            for j, key in enumerate(data.attrs['keys']):
                data[i, j] = antenna_metadata.pop(bytes.decode(key))
            antenna_metadatas.append(antenna_metadata)
        self._write_metadata("monte_carlo_data/antennas", antenna_metadatas)

    @property
    def has_detector(self):
        return hasattr(self, "_detector")


    def _write_particles(self, event):
        self._counters['events'] += 1
        start_index = self._counters['particles'] + 1
        self._counters['particles'] += len(event)
        metadata = self._create_metadataset("monte_carlo_data/particles")
        # Reshape metadata datasets to accomodate the event
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['particles']+1, axis=0)
        float_data.resize(self._counters['particles']+1, axis=0)

        self._write_indices("monte_carlo_data/particles",
                            start_index, len(event))
        self._write_metadata("monte_carlo_data/particles", event._metadata,
                             start_index)


    def _write_trigger(self, triggered, trigger_extras={}):
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        self._counters['triggers'] += 1
        start_index = self._counters['waveform_triggers'] + 1
        self._counters['waveform_triggers'] += max_waves

        trigger_data = self._create_dataset("data/triggers")
        extra_data = self._create_dataset("monte_carlo_data/triggers")
        trigger_data.resize(self._counters['triggers']+1, axis=0)
        extra_data.resize(self._counters['waveform_triggers']+1, axis=0)
        for key in trigger_extras:
            if key not in extra_data.attrs['keys']:
                extra_data.attrs['keys'] = np.append(extra_data.attrs['keys'], str.encode(key))
                extra_data.resize(extra_data.shape[1]+1, axis=1)

        trigger_data[self._counters['triggers']] = triggered

        for i, ant in enumerate(self._detector):
            # Store individual antenna triggers
            for j, wave in enumerate(ant.all_waveforms):
                extra_data[start_index+j, i] = triggered
            # Store extra triggers
            for key, val in trigger_extras.items():
                for k, match in enumerate(extra_data.attrs['keys']):
                    if key==match:
                        if isinstance(val, bool):
                            for jj in range(max_waves):
                                extra_data[start_index+jj, k] = val
                        else:
                            for jj in range(max_waves):
                                extra_data[start_index+jj, k] = val[jj]

        self._write_indices("data/triggers", self._counters['triggers'])
        self._write_indices("monte_carlo_data/triggers", start_index, max_waves)


    def _write_ray_data(self, ray_paths):
        if len(ray_paths)!=len(self._detector):
            raise ValueError("Ray paths length doesn't match detector size ("+
                             str(len(ray_paths))+"!="+
                             str(len(self._detector))+")")
        max_waves = max(len(paths) for paths in ray_paths)
        start_index = self._counters['rays'] + 1
        self._counters['rays'] += max_waves
        metadata = self._create_metadataset("monte_carlo_data/rays")
        # Reshape metadata datasets to accomodate the ray data of each solution
        str_data = metadata['str']
        float_data = metadata['float']
        str_data.resize(self._counters['rays']+1, axis=0)
        float_data.resize(self._counters['rays']+1, axis=0)

        for i in range(max_waves):
            ray_metadata = []
            for paths in ray_paths:
                if i<len(paths):
                    ray_metadata.append(paths[i]._metadata)
                else:
                    ray_metadata.append({})
            self._write_metadata("monte_carlo_data/rays", ray_metadata,
                                 start_index+i)

        self._write_indices("monte_carlo_data/rays", start_index, max_waves)


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
        data = self._create_dataset("monte_carlo_data/noise")
        data.resize(self._counters['noise']+1, axis=0)

        self._write_indices("monte_carlo_data/noise", self._counters['noise'])
        for i, ant in enumerate(self._detector):
            data[self._counters['noise'], i] = np.array(self._get_noise_bases(ant))


    def _write_waveforms(self):
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        start_index = self._counters['waveforms'] + 1
        self._counters['waveforms'] += max_waves

        data = self._create_dataset("data/waveforms")
        data.resize(self._counters['waveforms']+1, axis=0)

        for i, ant in enumerate(self._detector):
            for j, wave in enumerate(ant.all_waveforms):
                data[start_index+j, i] = np.array([wave.times, wave.values])

        self._write_indices("data/waveforms", start_index, max_waves)


    def add(self, event, triggered=None, ray_paths=None,
            trigger_extras={}):
        if self.verbosity in self._ray_verbosities and ray_paths is None:
            raise ValueError("Ray path information must be provided for "+
                             "verbosity level "+str(self.verbosity))
        if self.verbosity in self._trig_only_verbosities:
            if triggered is None:
                raise ValueError("Trigger information must be provided for "+
                                 "verbosity level "+str(self.verbosity))
            if self.verbosity in self._event_verbosities:
                if self.verbosity!=Verbosity.event_data_triggered_only or triggered:
                    self._write_particles(event)
                    self._write_trigger(triggered, trigger_extras)
            if triggered:
                if self.verbosity in self._ray_verbosities:
                    self._write_ray_data(ray_paths)
                if self.verbosity in self._noise_verbosities:
                    self._write_noise_data()
                if self.verbosity in self._wave_verbosities:
                    self._write_waveforms()
        else:
            if self.verbosity==Verbosity.event_data and triggered is None:
                raise ValueError("Trigger information must be provided for "+
                                 "verbosity level "+str(self.verbosity))
            if self.verbosity in self._event_verbosities:
                self._write_particles(event)
                if triggered is not None:
                    self._write_trigger(triggered, trigger_extras)
            if self.verbosity in self._ray_verbosities:
                self._write_ray_data(ray_paths)
            if self.verbosity in self._noise_verbosities:
                self._write_noise_data()
            if self.verbosity in self._wave_verbosities:
                self._write_waveforms()


    def add_metadata(self, file_metadata):
        self._write_metadata("file_metadata", file_metadata)



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
