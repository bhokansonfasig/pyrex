"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

from enum import Enum
import h5py
import numpy as np
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
    reconstruction_data_triggered_only = 3
    waveforms_triggered_only = 3
    reconstruction_data = 4
    waveforms = 4
    all_data_triggered_only = 5
    comlete_triggered_only = 5
    all_data = 6
    complete = 6
    maximum = 6

    default = 1


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
        self.verbosity = get_from_enum(verbosity, Verbosity)
        if self.verbosity not in accepted_verbosities:
            raise ValueError("Unable to write with verbosity "+
                             self.verbosity)


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
            Verbosity.simulation_data_triggered_only,
            Verbosity.simulation_data,
            Verbosity.reconstruction_data_triggered_only,
            Verbosity.reconstruction_data,
            Verbosity.all_data_triggered_only,
            Verbosity.all_data
        ]
        self._set_verbosity_level(verbosity, accepted_verbosities)

    def open(self):
        self._file = h5py.File(self.filename, mode='w')
        self._file.create_group("metadata")
        self._file['metadata'].create_group("events")
        self._file['metadata'].create_group("antennas")
        self._file['metadata'].create_group("waveforms")
        self._file.create_group("analysis")
        self._file['metadata']['events'].create_dataset(
            name="str", shape=(0, 0, 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None, None, None)
        )
        self._file['metadata']['events'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        self._file['metadata']['events'].create_dataset(
            name="float", shape=(0, 0, 0),
            dtype=np.float_, maxshape=(None, None, None)
        )
        self._file['metadata']['events'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        self._counter = 0

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
                elif isinstance(val, (int, float)):
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
                    raise ValueError("Must be str, int, or float")

    def set_detector(self, detector):
        self._detector = detector
        waveform_type = h5py.special_dtype(vlen=np.float_)
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
        # Complex events may have more than two waveforms per antenna,
        # store them separately to keep files from bloating with zeros
        complex_data = self._file.create_dataset(
            name="complex_events", shape=(0, len(detector), 2, 2),
            dtype=waveform_type, maxshape=(None, len(detector), None, 2)
        )
        complex_data.dims[0].label = "events"
        complex_data.dims[1].label = "antennas"
        complex_data.dims[2].label = "waveforms"

        self._file['metadata']['antennas'].create_dataset(
            name="str", shape=(len(detector), 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(len(detector), None)
        )
        self._file['metadata']['antennas'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        self._file['metadata']['antennas'].create_dataset(
            name="float", shape=(len(detector), 0),
            dtype=np.float_, maxshape=(len(detector), None)
        )
        self._file['metadata']['antennas'].create_dataset(
            name="float_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )

        # Each detector/antenna should write its own metadata, but for now
        # let's just capture a couple general-purpose attributes
        antenna_metadata = []
        for ant in detector:
            antenna_metadata.append(
                {
                    "antenna_class": str(type(ant)),
                    "x-position": ant.position[0],
                    "y-position": ant.position[1],
                    "z-position": ant.position[2]
                }
            )

        self._write_metadata(antenna_metadata, 'antennas')


    def _write_particles(self, event):
        # Event/Particle class should write its own metadata, but for now
        # let's just capture a couple general-purpose attributes
        particle_metadata = []
        for particle in event:
            particle_metadata.append(
                {
                    "particle_id": str(particle.id),
                    "x-position": float(particle.vertex[0]),
                    "y-position": float(particle.vertex[1]),
                    "z-position": float(particle.vertex[2]),
                    "x-direction": float(particle.direction[0]),
                    "y-direction": float(particle.direction[1]),
                    "z-direction": float(particle.direction[2]),
                    "energy": float(particle.energy)
                }
            )

        str_data = self._file['metadata']['events']['str']
        float_data = self._file['metadata']['events']['float']
        str_data.resize(self._counter, axis=0)
        float_data.resize(self._counter, axis=0)
        str_data.resize(max(len(event), str_data.shape[1]), axis=1)
        float_data.resize(max(len(event), str_data.shape[1]), axis=1)

        self._write_metadata(particle_metadata, 'events', self._counter-1)


    def _write_waveforms(self):
        data = self._file['events']
        data.resize(self._counter, axis=0)
        for i, ant in enumerate(self._detector):
            for j, wave in enumerate(ant.all_waveforms):
                data[self._counter-1, i, j] = np.array([wave.times, wave.values])


    def add(self, event, triggered=None):
        self._counter += 1
        self._write_particles(event)
        if self.verbosity==Verbosity.events_only:
            return
        if 'triggered_only' in self.verbosity.name:
            if triggered is None:
                self._counter -= 1
                raise ValueError("Trigger information must be provided")
            if triggered:
                self._write_waveforms()
        else:
            self._write_waveforms()


    # def add_metadata(self, file_metadata):
    #     self._file['metadata'].attrs = file_metadata



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
                'particle_id', 'x-position', 'y-position', 'z-position',
                'x-direction', 'y-direction', 'z-direction', 'energy'
            ]
            for key in required_keys:
                if key not in particle_metadata:
                    raise ValueError("Event metadata does not have a value for "+key)
            particle_id = particle_metadata['particle_id'][5:]
            vertex = (particle_metadata['x-position'],
                      particle_metadata['y-position'],
                      particle_metadata['z-position'])
            direction = (particle_metadata['x-direction'],
                         particle_metadata['y-direction'],
                         particle_metadata['z-direction'])
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
            position = (antenna_metadata.pop('x-position'),
                        antenna_metadata.pop('y-position'),
                        antenna_metadata.pop('z-position'))
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
