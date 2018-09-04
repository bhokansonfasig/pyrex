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
        self._file.create_group("events")
        self._file.create_group("complex_events")
        self._file.create_group("metadata")
        self._file['metadata'].create_group("events")
        self._file['metadata'].create_group("antennas")
        self._file['metadata'].create_group("waveforms")
        self._file.create_group("analysis")
        str_data = self._file['metadata']['events'].create_dataset(
            name="str", shape=(0, 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(None, None)
        )
        str_data.attrs['row_names'] = []
        float_data = self._file['metadata']['events'].create_dataset(
            name="float", shape=(0, 0),
            dtype=np.float_, maxshape=(None, None)
        )
        float_data.attrs['row_names'] = []
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
        data = self._file['events'].create_dataset(
            name="data", shape=(0, len(detector), 2, 2),
            dtype=waveform_type, maxshape=(None, len(detector), 2, 2)
        )
        data.dims[0].label = "events"
        data.dims[1].label = "antennas"
        data.dims[2].label = "waveforms"
        # Complex events may have more than two waveforms per antenna,
        # store them separately to keep files from bloating with zeros
        complex_data = self._file['complex_events'].create_dataset(
            name="data", shape=(0, len(detector), 2, 2),
            dtype=waveform_type, maxshape=(None, len(detector), None, 2)
        )
        complex_data.dims[0].label = "events"
        complex_data.dims[1].label = "antennas"
        complex_data.dims[2].label = "waveforms"

        str_data = self._file['metadata']['antennas'].create_dataset(
            name="str", shape=(len(detector), 0),
            dtype=h5py.special_dtype(vlen=str), maxshape=(len(detector), None)
        )
        str_data.attrs['row_names'] = []
        float_data = self._file['metadata']['antennas'].create_dataset(
            name="float", shape=(len(detector), 0),
            dtype=np.float_, maxshape=(len(detector), None)
        )
        float_data.attrs['row_names'] = []

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
                    if key not in list(str_data.attrs['row_names']):
                        j = len(str_data.attrs['row_names'])
                        np.append(str_data.attrs['row_names'], key)
                        str_data.resize(j+1, axis=1)
                    else:
                        j = str_data.attrs['row_names'].index(key)
                    str_data[i][j] = val
                elif isinstance(val, (int, float)):
                    if key not in list(float_data.attrs['row_names']):
                        j = len(float_data.attrs['row_names'])
                        np.append(float_data.attrs['row_names'], key)
                        float_data.resize(j+1, axis=1)
                    else:
                        j = float_data.attrs['row_names'].index(key)
                    float_data[i][j] = val
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
        float_data = self._file['metadata']['events']['float']
        str_data.resize(len(event), axis=0)
        float_data.resize(len(event), axis=0)

        for i, meta in enumerate(particle_metadata):
            for key, val in meta.items():
                if isinstance(val, str):
                    if key not in list(str_data.attrs['row_names']):
                        j = len(str_data.attrs['row_names'])
                        np.append(str_data.attrs['row_names'], key)
                        str_data.resize(j+1, axis=1)
                    else:
                        j = str_data.attrs['row_names'].index(key)
                    str_data[i][j] = val
                elif isinstance(val, (int, float)):
                    if key not in list(float_data.attrs['row_names']):
                        j = len(float_data.attrs['row_names'])
                        np.append(float_data.attrs['row_names'], key)
                        float_data.resize(j+1, axis=1)
                    else:
                        j = float_data.attrs['row_names'].index(key)
                    float_data[i][j] = val
                else:
                    raise ValueError("Must be str, int, or float ("+key+", "+val+")")


    def _write_waveforms(self):
        data = self._file['events']['data']
        data.resize(self._counter, axis=0)
        for i, ant in enumerate(self._detector):
            for j, wave in enumerate(ant.all_waveforms):
                data[self._counter-1, i, j] = np.array([wave.times, wave.values])


    def add(self, event, triggered=None):
        self._counter += 1
        if self.verbosity==Verbosity.events_only:
            self._write_particles(event)
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
