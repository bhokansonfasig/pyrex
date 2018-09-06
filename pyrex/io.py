"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

from enum import Enum
import datetime
import h5py
import numpy as np
from pyrex.__about__ import __version__
from pyrex.internal_functions import get_from_enum
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.particle import Particle, Event


class Verbosity(Enum):
    events_only = 0
    minimum = 0
    basic = 0
    simulation_data_triggered_only = 1
    reproducible_triggered_only = 1
    simulation_data = 2
    reproducible = 2
    ray_data_triggered_only = 3
    debugging_triggered_only = 3
    ray_data = 4
    debugging = 4
    reconstruction_data_triggered_only = 5
    waveforms_triggered_only = 5
    reconstruction_data = 6
    waveforms = 6
    all_data_triggered_only = 7
    comlete_triggered_only = 7
    all_data = 8
    complete = 8
    maximum = 8

    default = 3


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


class BaseWriter:
    def __init__(self, filename, verbosity=Verbosity.default):
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

    def _set_verbosity_level(self, verbosity, accepted_verbosities):
        if verbosity==-1:
            verbosity = "maximum"
        self.verbosity = get_from_enum(verbosity, Verbosity)
        if self.verbosity not in accepted_verbosities:
            raise ValueError("Unable to write file with verbosity level '"+
                             str(verbosity)+"'")


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




class HDF5Writer(BaseWriter):
    def __init__(self, filename, verbosity=Verbosity.default):
        if filename.endswith(".hdf5") or filename.endswith(".h5"):
            self.filename = filename
        else:
            self.filename = filename+".h5"
        accepted_verbosities = [
            Verbosity.events_only,
            # Verbosity.simulation_data_triggered_only,
            # Verbosity.simulation_data,
            Verbosity.ray_data_triggered_only,
            Verbosity.ray_data,
            Verbosity.reconstruction_data_triggered_only,
            Verbosity.reconstruction_data,
            # Verbosity.all_data_triggered_only,
            # Verbosity.all_data
        ]
        self._set_verbosity_level(verbosity, accepted_verbosities)

    def open(self):
        self._file = h5py.File(self.filename, mode='w')
        self._file.create_group("analysis")
        # Set up metadata groups
        self._file.create_group("metadata")
        self._file['metadata'].create_group("file")
        self._file['metadata'].create_group("events")
        self._file['metadata'].create_group("antennas")
        self._file['metadata'].create_group("rays")
        self._file['metadata'].create_group("waveforms")
        self._file.create_group("data_indices")

        # Set up trigger dataset
        self._file.create_dataset(
            name="triggers", shape=(0,),
            dtype=np.bool_, maxshape=(None,)
        )

        # Set up event index datasets
        self._file['data_indices'].create_dataset(
            name="events", shape=(0,),
            dtype=np.int_, maxshape=(None,)
        )
        self._file['data_indices'].create_dataset(
            name="complex_events", shape=(0,),
            dtype=np.int_, maxshape=(None,)
        )
        event_meta_indices = self._file['data_indices'].create_dataset(
            name="event_metadata", shape=(0,),
            dtype=np.int_, maxshape=(None,)
        )

        # Set up file metadata datasets
        str_data = self._file['metadata']['file'].create_dataset(
            name="str", shape=(1, 0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(1, None,)
        )
        str_keys = self._file['metadata']['file'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[0].attach_scale(str_keys)
        float_data = self._file['metadata']['file'].create_dataset(
            name="float", shape=(1, 0,),
            dtype=np.float_, maxshape=(1, None,)
        )
        float_keys = self._file['metadata']['file'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[0].attach_scale(float_keys)

        # Set up event metadata datasets
        str_data = self._file['metadata']['events'].create_dataset(
            name="str", shape=(0, 0, 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None, None, None)
        )
        str_keys = self._file['metadata']['events'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims[0].label = "events"
        str_data.dims[1].label = "particles"
        str_data.dims[2].label = "attributes"
        str_data.dims.create_scale(event_meta_indices, 'event_numbers')
        str_data.dims[0].attach_scale(event_meta_indices)
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[2].attach_scale(str_keys)
        float_data = self._file['metadata']['events'].create_dataset(
            name="float", shape=(0, 0, 0),
            dtype=np.float_, maxshape=(None, None, None)
        )
        float_keys = self._file['metadata']['events'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims[0].label = "events"
        float_data.dims[1].label = "particles"
        float_data.dims[2].label = "attributes"
        float_data.dims.create_scale(event_meta_indices, 'event_numbers')
        float_data.dims[0].attach_scale(event_meta_indices)
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[2].attach_scale(float_keys)

        # Set event number counters
        self._event_counter = -1
        self._ray_counter = -1
        self._wave_counter = -1
        self._complex_wave_counter = -1

        # Write some generic metadata about the file production
        major, minor, patch = __version__.split('.')
        now = datetime.datetime.now()
        metadata = {
            "file_version": "1.0",
            "file_version_major": 1,
            "file_version_minor": 0,
            "pyrex_version": __version__,
            "pyrex_version_major": int(major),
            "pyrex_version_minor": int(minor),
            "pyrex_version_patch": int(patch),
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
                    if isinstance(val[0], str):
                        val_type = "string"
                    elif np.isscalar(val[0]):
                        val_type = "float"
                    else:
                        raise ValueError(key+" value ("+str(val[0])+") must be"+
                                         " string or scalar")
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
        waveform_type = h5py.special_dtype(vlen=np.float_)
        # Set up event and complex_event datasets
        # Dimensions:
        #   0 - Number of events
        #   1 - Number of antennas
        #   2 - Number of waveforms per antenna (usually 2: direct & reflected)
        #   3 - Number of value types (2: times & values)
        data = self._file.create_dataset(
            name="events", shape=(0, len(detector), 2, 2),
            dtype=waveform_type, maxshape=(None, len(detector), 2, 2)
        )
        data.dims[0].label = "events"
        data.dims[1].label = "antennas"
        data.dims[2].label = "waveforms"
        data.dims.create_scale(self._file['data_indices']['events'], 'event_numbers')
        data.dims[0].attach_scale(self._file['data_indices']['events'])

        # Complex events may have more than two waveforms per antenna,
        # store them separately to keep files from bloating with zeros
        complex_data = self._file.create_dataset(
            name="complex_events", shape=(0, len(detector), 0, 2),
            dtype=waveform_type, maxshape=(None, len(detector), None, 2)
        )
        complex_data.dims[0].label = "events"
        complex_data.dims[1].label = "antennas"
        complex_data.dims[2].label = "waveforms"
        data.dims.create_scale(self._file['data_indices']['complex_events'], 'event_numbers')
        data.dims[0].attach_scale(self._file['data_indices']['complex_events'])

        # Set up antenna metadata datasets
        str_data = self._file['metadata']['antennas'].create_dataset(
            name="str", shape=(len(detector), 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(len(detector), None)
        )
        str_keys = self._file['metadata']['antennas'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims[0].label = "antennas"
        str_data.dims[1].label = "attributes"
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[1].attach_scale(str_keys)
        float_data = self._file['metadata']['antennas'].create_dataset(
            name="float", shape=(len(detector), 0),
            dtype=np.float_, maxshape=(len(detector), None)
        )
        float_keys = self._file['metadata']['antennas'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims[0].label = "antennas"
        float_data.dims[1].label = "attributes"
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[1].attach_scale(float_keys)

        # Write antenna metadata
        self._write_metadata([antenna._metadata for antenna in detector],
                             'antennas')

        # Set up ray metadata datasets
        str_data = self._file['metadata']['rays'].create_dataset(
            name="str", shape=(0, len(detector), 0, 0),
            dtype=h5py.special_dtype(vlen=str),
            maxshape=(None, len(detector), None, None)
        )
        str_keys = self._file['metadata']['rays'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims[0].label = "events"
        str_data.dims[1].label = "antennas"
        str_data.dims[2].label = "attributes"
        str_data.dims[3].label = "solutions"
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[2].attach_scale(str_keys)
        float_data = self._file['metadata']['rays'].create_dataset(
            name="float", shape=(0, len(detector), 0, 0),
            dtype=np.float_, maxshape=(None, len(detector), None, None)
        )
        float_keys = self._file['metadata']['rays'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims[0].label = "events"
        float_data.dims[1].label = "antennas"
        float_data.dims[2].label = "attributes"
        float_data.dims[3].label = "solutions"
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[2].attach_scale(float_keys)

        # Set up waveform metadata datasets
        str_data = self._file['metadata']['waveforms'].create_dataset(
            name="str", shape=(0, len(detector), 0, 0),
            dtype=h5py.special_dtype(vlen=str),
            maxshape=(None, len(detector), None, None)
        )
        str_keys = self._file['metadata']['waveforms'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        str_data.dims[0].label = "events"
        str_data.dims[1].label = "antennas"
        str_data.dims[2].label = "attributes"
        str_data.dims[3].label = "waveforms"
        str_data.dims.create_scale(str_keys, 'attribute_names')
        str_data.dims[2].attach_scale(str_keys)
        float_data = self._file['metadata']['waveforms'].create_dataset(
            name="float", shape=(0, len(detector), 0, 0),
            dtype=np.float_, maxshape=(None, len(detector), None, None)
        )
        float_keys = self._file['metadata']['waveforms'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data.dims[0].label = "events"
        float_data.dims[1].label = "antennas"
        float_data.dims[2].label = "attributes"
        float_data.dims[3].label = "waveforms"
        float_data.dims.create_scale(float_keys, 'attribute_names')
        float_data.dims[2].attach_scale(float_keys)


    def _write_particles(self, event):
        self._event_counter += 1
        str_data = self._file['metadata']['events']['str']
        float_data = self._file['metadata']['events']['float']
        str_data.resize(self._event_counter+1, axis=0)
        float_data.resize(self._event_counter+1, axis=0)
        str_data.resize(max(len(event), str_data.shape[1]), axis=1)
        float_data.resize(max(len(event), str_data.shape[1]), axis=1)

        self._write_event_number('event_metadata', self._event_counter)

        self._write_metadata(event._metadata, 'events', self._event_counter)


    def _write_trigger(self, triggered):
        trigger_data = self._file['triggers']
        trigger_data.resize(self._event_counter+1, axis=0)
        trigger_data[self._event_counter] = bool(triggered)


    def _write_ray_data(self, ray_paths):
        self._ray_counter += 1
        if len(ray_paths)!=len(self._detector):
            raise ValueError("Ray paths length doesn't match detector ("+
                             str(len(ray_paths))+"!="+
                             str(len(self._detector))+")")
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

        self._write_metadata(ray_metadata, 'rays', self._ray_counter)


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

        data = self._file[dataset_name]
        data.resize(index+1, axis=0)
        self._write_event_number(dataset_name, index)

        str_data = self._file['metadata']['waveforms']['str']
        float_data = self._file['metadata']['waveforms']['float']
        str_data.resize(self._wave_counter+self._complex_wave_counter+2, axis=0)
        float_data.resize(self._wave_counter+self._complex_wave_counter+2, axis=0)
        str_data.resize(max(max_waves, str_data.shape[3]), axis=3)
        float_data.resize(max(max_waves, float_data.shape[3]), axis=3)

        waveform_metadata = []
        for i, ant in enumerate(self._detector):
            meta = {
                "triggered": [0]*max_waves
            }
            for j, wave in enumerate(ant.all_waveforms):
                data[index, i, j] = np.array([wave.times, wave.values])
                meta['triggered'][j] = int(ant.trigger(wave))

        self._write_metadata(waveform_metadata, 'waveforms',
                             self._wave_counter+self._complex_wave_counter+1)


    def add(self, event, triggered=None, ray_paths=None):
        if self.verbosity==Verbosity.events_only:
            if triggered is None:
                raise ValueError("Trigger information must be provided for "+
                                 "verbosity level "+str(self.verbosity))
            self._write_particles(event)
            self._write_trigger(triggered)
        elif 'triggered_only' in self.verbosity.name:
            if triggered is None:
                raise ValueError("Trigger information must be provided for "+
                                 "verbosity level "+str(self.verbosity))
            self._write_particles(event)
            self._write_trigger(triggered)
            if triggered:
                if self.verbosity.value>=Verbosity.ray_data_triggered_only.value:
                    self._write_ray_data(ray_paths)
                if self.verbosity.value>=Verbosity.reconstruction_data_triggered_only.value:
                    self._write_waveforms()
        else:
            self._write_particles(event)
            if triggered is not None:
                self._write_trigger(triggered)
            if self.verbosity.value>=Verbosity.ray_data.value:
                self._write_ray_data(ray_paths)
            if self.verbosity.value>=Verbosity.reconstruction_data.value:
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
