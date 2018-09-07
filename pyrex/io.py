"""
Module containing classes for reading and writing data files.

Includes reader and writer for hdf5 data files, as well as base reader and
writer classes which can be extended to read and write other file formats.

"""

from enum import Enum
import h5py
import numpy as np
from pyrex.internal_functions import get_from_enum


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
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class BaseWriter:
    def __init__(self, filename, verbosity=Verbosity.default):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def _set_verbosity_level(self, verbosity, accepted_verbosities):
        self.verbosity = get_from_enum(verbosity, Verbosity)
        if self.verbosity not in accepted_verbosities:
            raise ValueError("Unable to write with verbosity "+
                             self.verbosity)


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

        str_data = self._file['metadata']['antennas'].create_dataset(
            name="str", shape=(len(detector), 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(len(detector), None)
        )
        str_keys = self._file['metadata']['antennas'].create_dataset(
            name="str_keys", shape=(0,),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None,)
        )
        float_data = self._file['metadata']['antennas'].create_dataset(
            name="float", shape=(len(detector), 0),
            dtype=np.float_, maxshape=(len(detector), None)
        )
        float_keys = self._file['metadata']['antennas'].create_dataset(
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

        for i, meta in enumerate(antenna_metadata):
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
                        str_data.resize(j+1, axis=1)
                    str_data[i,j] = val
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
                        float_data.resize(j+1, axis=1)
                    float_data[i,j] = val
                else:
                    raise ValueError("Must be str, int, or float")


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
                }
            )

        str_data = self._file['metadata']['events']['str']
        str_keys = self._file['metadata']['events']['str_keys']
        float_data = self._file['metadata']['events']['float']
        float_keys = self._file['metadata']['events']['float_keys']
        str_data.resize(self._counter, axis=0)
        float_data.resize(self._counter, axis=0)
        str_data.resize(max(len(event), str_data.shape[1]), axis=1)
        float_data.resize(max(len(event), str_data.shape[1]), axis=1)

        for i, meta in enumerate(particle_metadata):
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
                        str_data.resize(j+1, axis=2)
                    str_data[self._counter-1, i, j] = val
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
                        float_data.resize(j+1, axis=2)
                    float_data[self._counter-1, i, j] = val
                else:
                    raise ValueError("Must be str, int, or float")


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
