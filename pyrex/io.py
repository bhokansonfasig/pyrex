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
        # Create basic groups
        self._file.create_group("analysis")
        self._file.create_group("metadata")
        self._file.create_group("data_indices")

        # Set event number counters
        self._event_counter = -1
        self._trigger_counter = -1
        self._ray_counter = -1
        self._noise_counter = -1
        self._wave_counter = -1
        self._complex_wave_counter = -1

        # Write some generic metadata about the file production
        self._create_metadataset("file")
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
        self._write_metadata([metadata], 'file')

    def close(self):
        self._file.close()
        self._is_open = False

    @property
    def is_open(self):
        return self._is_open


    def _create_dataset(self, name):
        if name=="events":
            # Dimensions:
            #   0 - Number of events
            #   1 - Number of antennas
            #   2 - Number of waveforms per antenna (2: direct & reflected)
            #   3 - Number of value types (2: times & values)
            data = self._file.create_dataset(
                name="events", shape=(0, len(self._detector), 2, 2),
                dtype=h5py.special_dtype(vlen=np.float_),
                maxshape=(None, len(self._detector), 2, 2)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "antennas"
            data.dims[2].label = "waveforms"
            if "events" in self._file['data_indices']:
                indices = self._file['data_indices']['events']
            else:
                indices = self._create_dataset("data_indices/events")
            data.dims.create_scale(indices, 'event_numbers')
            data.dims[0].attach_scale(indices)

        elif name=="complex_events":
            # Complex events may have more than two waveforms per antenna,
            # store them separately to keep files from bloating with zeros
            data = self._file.create_dataset(
                name="complex_events", shape=(0, len(self._detector), 0, 2),
                dtype=h5py.special_dtype(vlen=np.float_),
                maxshape=(None, len(self._detector), None, 2)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "antennas"
            data.dims[2].label = "waveforms"
            if "complex_events" in self._file['data_indices']:
                indices = self._file['data_indices']['complex_events']
            else:
                indices = self._create_dataset("data_indices/complex_events")
            data.dims.create_scale(indices, 'event_numbers')
            data.dims[0].attach_scale(indices)

        elif name=="triggers":
            data = self._file.create_dataset(
                name="triggers", shape=(0,),
                dtype=np.bool_, maxshape=(None,)
            )
            if "triggers" in self._file['data_indices']:
                indices = self._file['data_indices']['triggers']
            else:
                indices = self._create_dataset("data_indices/triggers")
            data.dims.create_scale(indices, 'event_numbers')
            data.dims[0].attach_scale(indices)

        elif name=="noise_bases":
            data = self._file.create_dataset(
                name="noise_bases", shape=(0, len(self._detector), 3),
                dtype=h5py.special_dtype(vlen=np.float_),
                maxshape=(None, len(self._detector), 3)
            )
            data.dims[0].label = "events"
            data.dims[1].label = "antennas"
            if "noise_bases" in self._file['data_indices']:
                indices = self._file['data_indices']['noise_bases']
            else:
                indices = self._create_dataset("data_indices/noise_bases")
            data.dims.create_scale(indices, 'event_numbers')
            data.dims[0].attach_scale(indices)

        elif name.startswith("data_indices/"):
            index_name = name[13:]
            data = self._file['data_indices'].create_dataset(
                name=index_name, shape=(0,),
                dtype=np.int_, maxshape=(None,)
            )

        else:
            raise ValueError("Unrecognized dataset name '"+name+"'")

        return data


    def _create_metadataset(self, name):
        # Don't recreate if it already exists
        if name in self._file['metadata']:
            return

        if name=="file":
            shape = (1, 0)
            maxshape = (1, None)
            key_dim = 1
            def apply_labels(data):
                data.dims[1].label = "attributes"
        elif name=="events":
            shape = (0, 0, 0)
            maxshape = (None, None, None)
            key_dim = 2
            def apply_labels(data):
                data.dims[0].label = "events"
                data.dims[1].label = "particles"
                data.dims[2].label = "attributes"
        elif name=="antennas":
            shape = (len(self._detector), 0)
            maxshape = (len(self._detector), None)
            key_dim = 1
            def apply_labels(data):
                data.dims[0].label = "antennas"
                data.dims[1].label = "attributes"
        elif name=="rays":
            shape = (0, len(self._detector), 0, 0)
            maxshape = (None, len(self._detector), None, None)
            key_dim = 2
            def apply_labels(data):
                data.dims[0].label = "events"
                data.dims[1].label = "antennas"
                data.dims[2].label = "attributes"
                data.dims[3].label = "solutions"
        elif name=="waveforms":
            shape = (0, len(self._detector), 0, 0)
            maxshape = (None, len(self._detector), None, None)
            key_dim = 2
            def apply_labels(data):
                data.dims[0].label = "events"
                data.dims[1].label = "antennas"
                data.dims[2].label = "attributes"
                data.dims[3].label = "waveforms"
        else:
            raise ValueError("Unrecognized metadata name '"+name+"'")

        # Create group with the given name, create string and float datasets,
        # link attribute names, and label the dimensions
        self._file['metadata'].create_group(name)
        str_data = self._file['metadata'][name].create_dataset(
            name="str", shape=shape,
            dtype=h5py.special_dtype(vlen=str), maxshape=maxshape
        )
        str_keys = self._file['metadata'][name].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[key_dim].attach_scale(str_keys)
        apply_labels(str_data)
        float_data = self._file['metadata'][name].create_dataset(
            name="float", shape=shape,
            dtype=np.float_, maxshape=maxshape
        )
        float_keys = self._file['metadata'][name].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[key_dim].attach_scale(float_keys)
        apply_labels(float_data)

        # Link event numbers for the relevant datasets
        if name in ["events", "rays", "waveforms"]:
            meta_name = "metadata_"+name
            if meta_name in self._file['data_indices']:
                indices = self._file['data_indices'][meta_name]
            else:
                indices = self._create_dataset("data_indices/"+meta_name)
            str_data.dims.create_scale(indices, 'event_numbers')
            str_data.dims[0].attach_scale(indices)
            float_data.dims.create_scale(indices, 'event_numbers')
            float_data.dims[0].attach_scale(indices)


    def _write_metadata(self, metadata, group, index=None):
        str_data = self._file['metadata'][group]['str']
        str_keys = self._file['metadata'][group]['str_keys']
        float_data = self._file['metadata'][group]['float']
        float_keys = self._file['metadata'][group]['float_keys']
        data_axis = 1 if index is None else 2
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
                    j = -1
                    for k, match in enumerate(str_keys[:]):
                        if match==key:
                            j = k
                            break
                    if j==-1:
                        j = str_keys.size
                        str_keys.resize(j+1, axis=0)
                        str_keys[j] = key
                        str_data.resize(j+1, axis=data_axis)
                    if index is None:
                        str_data[i, j] = val
                    else:
                        str_data[index, i, j] = val
                elif val_type=="float":
                    j = -1
                    for k, match in enumerate(float_keys[:]):
                        if match==key:
                            j = k
                            break
                    if j==-1:
                        j = float_keys.size
                        float_keys.resize(j+1, axis=0)
                        float_keys[j] = key
                        float_data.resize(j+1, axis=data_axis)
                    if index is None:
                        float_data[i, j] = val
                    else:
                        float_data[index, i, j] = val
                else:
                    raise ValueError("Unrecognized val_type")


    def _write_event_number(self, index_dataset_name, index):
        index_dataset = self._file['data_indices'][index_dataset_name]
        index_dataset.resize(index+1, axis=0)
        index_dataset[index] = self._event_counter


    def set_detector(self, detector):
        self._detector = detector
        # Write antenna metadata
        self._create_metadataset("antennas")
        self._write_metadata([antenna._metadata for antenna in detector],
                             'antennas')

    @property
    def has_detector(self):
        return hasattr(self, "_detector")


    def _write_particles(self, event):
        self._event_counter += 1
        if "events" not in self._file['metadata']:
            self._create_metadataset("events")
        # Reshape metadata datasets to accomodate the event and its particles
        str_data = self._file['metadata']['events']['str']
        float_data = self._file['metadata']['events']['float']
        str_data.resize(self._event_counter+1, axis=0)
        float_data.resize(self._event_counter+1, axis=0)
        str_data.resize(max(len(event), str_data.shape[1]), axis=1)
        float_data.resize(max(len(event), str_data.shape[1]), axis=1)

        self._write_event_number('metadata_events', self._event_counter)
        self._write_metadata(event._metadata, 'events', self._event_counter)


    def _write_trigger(self, triggered):
        self._trigger_counter += 1
        if "triggers" in self._file:
            trigger_data = self._file['triggers']
        else:
            trigger_data = self._create_dataset("triggers")
        trigger_data.resize(self._trigger_counter+1, axis=0)

        self._write_event_number('triggers', self._trigger_counter)
        trigger_data[self._trigger_counter] = bool(triggered)


    def _write_ray_data(self, ray_paths):
        self._ray_counter += 1
        if len(ray_paths)!=len(self._detector):
            raise ValueError("Ray paths length doesn't match detector size ("+
                             str(len(ray_paths))+"!="+
                             str(len(self._detector))+")")
        if "rays" not in self._file['metadata']:
            self._create_metadataset("rays")
        # Reshape metadata datasets to accomodate the ray data of each solution
        max_waves = max(len(paths) for paths in ray_paths)
        str_data = self._file['metadata']['rays']['str']
        float_data = self._file['metadata']['rays']['float']
        str_data.resize(self._ray_counter+1, axis=0)
        float_data.resize(self._ray_counter+1, axis=0)
        str_data.resize(max(max_waves, str_data.shape[3]), axis=3)
        float_data.resize(max(max_waves, float_data.shape[3]), axis=3)

        ray_metadata = []
        for paths in ray_paths:
            meta = {}
            for j, path in enumerate(paths):
                for key, val in path._metadata.items():
                    if key not in meta:
                        if isinstance(val, str):
                            meta[key] = [""]*max_waves
                        elif np.isscalar(val):
                            meta[key] = [0]*max_waves
                        else:
                            raise ValueError(key+" value ("+str(val)+
                                             ") must be string or scalar")
                    meta[key][j] = val
            ray_metadata.append(meta)

        self._write_event_number('metadata_rays', self._ray_counter)
        self._write_metadata(ray_metadata, 'rays', self._ray_counter)


    def _get_noise_bases(self, antenna):
        while hasattr(antenna, "antenna"):
            antenna = antenna.antenna
        noise = antenna._noise_master
        if noise is None:
            return [], [], []
        else:
            return noise.freqs, noise.amps, noise.phases


    def _write_noise_data(self):
        self._noise_counter += 1
        if "noise_bases" in self._file:
            data = self._file['noise_bases']
        else:
            data = self._create_dataset("noise_bases")
        data.resize(self._noise_counter+1, axis=0)

        self._write_event_number('noise_bases', self._noise_counter)
        for i, ant in enumerate(self._detector):
            data[self._noise_counter, i] = np.array(self._get_noise_bases(ant))


    def _write_waveforms(self):
        max_waves = max(len(ant.all_waveforms) for ant in self._detector)
        if max_waves<=2:
            self._wave_counter += 1
            dataset_name = 'events'
            index = self._wave_counter
        else:
            self._complex_wave_counter += 1
            dataset_name = 'complex_events'
            index = self._complex_wave_counter

        if dataset_name in self._file:
            data = self._file[dataset_name]
        else:
            data = self._create_dataset(dataset_name)
        data.resize(index+1, axis=0)
        if "waveforms" not in self._file['metadata']:
            self._create_metadataset("waveforms")
        # Reshape metadata datasets to accomodate the metadata of each waveform
        str_data = self._file['metadata']['waveforms']['str']
        float_data = self._file['metadata']['waveforms']['float']
        str_data.resize(self._wave_counter+self._complex_wave_counter+2, axis=0)
        float_data.resize(self._wave_counter+self._complex_wave_counter+2, axis=0)
        str_data.resize(max(max_waves, str_data.shape[3]), axis=3)
        float_data.resize(max(max_waves, float_data.shape[3]), axis=3)

        self._write_event_number(dataset_name, index)

        waveform_metadata = []
        for i, ant in enumerate(self._detector):
            meta = {
                "triggered": [-1]*max_waves
            }
            for j, wave in enumerate(ant.all_waveforms):
                data[index, i, j] = np.array([wave.times, wave.values])
                meta['triggered'][j] = int(ant.trigger(wave))
            waveform_metadata.append(meta)

        self._write_event_number("metadata_waveforms", self._wave_counter
                                 +self._complex_wave_counter+1)
        self._write_metadata(waveform_metadata, 'waveforms', self._wave_counter
                             +self._complex_wave_counter+1)


    def add(self, event, triggered=None, ray_paths=None):
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
                    self._write_trigger(triggered)
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
                    self._write_trigger(event)
            if self.verbosity in self._ray_verbosities:
                self._write_ray_data(ray_paths)
            if self.verbosity in self._noise_verbosities:
                self._write_noise_data()
            if self.verbosity in self._wave_verbosities:
                self._write_waveforms()


    def add_metadata(self, file_metadata):
        self._write_metadata([file_metadata], 'file')



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
