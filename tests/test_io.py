"""File containing tests of pyrex io module"""

import pytest

from pyrex.io import HDF5Base, HDF5Reader, HDF5Writer, File
from pyrex.signals import Signal
from pyrex.antenna import Antenna
from pyrex.particle import Particle, Event
from pyrex.ray_tracing import RayTracer

import h5py
import numpy as np



@pytest.fixture
def h5base():
    """Fixture for forming basic HDF5Base object"""
    return HDF5Base(1, 0)

class TestHDF5Base:
    """Tests for HDF5Base class"""
    def test_creation(self, h5base):
        """Test initialization of HDF5Base object"""
        assert h5base._file_version_major == 1
        assert h5base._file_version_minor == 0

    def test_dataset_locations(self, h5base):
        """Test the dataset locations of an HDF5Base object"""
        locations = h5base._dataset_locations()
        assert locations['file_meta'] == "/file_metadata"
        assert locations['indices'] == "/event_indices"
        assert locations['waveforms'] == "/data/waveforms"
        assert locations['triggers'] == "/data/triggers"
        assert locations['antennas'] == "/data/antennas"
        assert locations['particles_meta'] == "/monte_carlo_data/particles"
        assert locations['antennas_meta'] == "/monte_carlo_data/antennas"
        assert locations['rays_meta'] == "/monte_carlo_data/rays"
        assert locations['mc_triggers'] == "/monte_carlo_data/triggers"
        assert locations['noise'] == "/monte_carlo_data/noise"

    def test_generate_location_names(self, h5base, tmpdir):
        """Test that location names can be generated from a file"""
        tmpfile = h5py.File(str(tmpdir.join('tmpfile.h5')))
        tmpfile.create_group('additional_group')
        tmpfile.create_dataset('additional_group/new_dataset', data=(0, 1, 2))
        tmpfile.create_dataset('additional_group/duplicate', data=(3, 4, 5))
        tmpfile.create_dataset('some_other_group/duplicate', data=(6, 7, 8))
        tmpfile.create_group('metadata_group')
        tmpfile.create_dataset('metadata_group/str', data=[])
        tmpfile.create_dataset('metadata_group/float', data=(0,))
        all_locations = h5base._generate_location_names(
            tmpfile, h5base._dataset_locations()
        )
        assert all_locations['file_meta'] == "/file_metadata"
        assert all_locations['indices'] == "/event_indices"
        assert all_locations['waveforms'] == "/data/waveforms"
        assert all_locations['new_dataset'] == "/additional_group/new_dataset"
        assert (all_locations['additional_group_duplicate'] ==
                "/additional_group/duplicate")
        assert (all_locations['some_other_group_duplicate'] ==
                "/some_other_group/duplicate")
        assert all_locations['metadata_group'] == "/metadata_group"

    def test_analysis_location(self, h5base):
        """Test transformation of names to analysis location"""
        assert (h5base._analysis_location('new_dataset')
                == '/monte_carlo_data/analysis/new_dataset')
        assert (h5base._analysis_location('data/another_new_dataset')
                == '/data/analysis/another_new_dataset')
        assert (h5base._analysis_location('/monte_carlo_data/analysis/proper')
                == '/monte_carlo_data/analysis/proper')
        assert (h5base._analysis_location('nested/group/with/dataset')
                == '/monte_carlo_data/analysis/nested/group/with/dataset')

    def test_read_metadata_to_dicts(self, h5base, tmpdir):
        """Test behavior of the _read_metadata_to_dicts method"""
        metadata = {
            "color": ['Red', 'Blue', 'Green'],
            "number": [1, 2, 3],
            "zero": [0, 0, 0],
        }
        tmpfile = h5py.File(str(tmpdir.join('tmpfile.h5')))
        tmpfile.create_group('metadata_group')
        str_dataset = tmpfile.create_dataset(
            'metadata_group/str',
            data=[[str.encode(x)] for x in metadata['color']]
        )
        str_dataset.attrs['keys'] = [str.encode('color')]
        float_dataset = tmpfile.create_dataset(
            'metadata_group/float',
            data=np.array([metadata['number'], metadata['zero']]).T
        )
        float_dataset.attrs['keys'] = [str.encode('number'), str.encode('zero')]
        read = h5base._read_metadata_to_dicts(tmpfile, 'metadata_group')
        for i, row in enumerate(read):
            assert row['color'] == str.encode(metadata['color'][i])
            assert row['number'] == metadata['number'][i]
            assert row['zero'] == metadata['zero'][i]
        read = h5base._read_metadata_to_dicts(tmpfile, 'metadata_group',
                                              str_keys=['COLOR'],
                                              float_keys=['NUMBER', 'ZERO'])
        for i, row in enumerate(read):
            assert row['COLOR'] == str.encode(metadata['color'][i])
            assert row['NUMBER'] == metadata['number'][i]
            assert row['ZERO'] == metadata['zero'][i]

    def test_read_dataset_to_dicts(self, h5base, tmpdir):
        """Test behavior of the _read_dataset_to_dicts method"""
        numbers = [1, 2, 3, 4, 5]
        zeros = [0, 0, 0, 0, 0]
        tmpfile = h5py.File(str(tmpdir.join('tmpfile.h5')))
        dataset = tmpfile.create_dataset(
            'float_data',
            data=np.array([numbers, zeros]).T
        )
        dataset.attrs['keys'] = [str.encode('number'), str.encode('zero')]
        read = h5base._read_dataset_to_dicts(tmpfile, 'float_data')
        for i, row in enumerate(read):
            assert row['number'] == numbers[i]
            assert row['zero'] == zeros[i]
        read = h5base._read_dataset_to_dicts(tmpfile, 'float_data',
                                             keys=['NUMBER', 'ZERO'])
        for i, row in enumerate(read):
            assert row['NUMBER'] == numbers[i]
            assert row['ZERO'] == zeros[i]

    def test_get_bool_dict(self, h5base, tmpdir):
        """Test creation of a boolean dictionary of existing datasets"""
        tmpfile = h5py.File(str(tmpdir.join('tmpfile.h5')))
        tmpfile.create_dataset('event_indices', data=(0, 1, 2, 3, 4))
        tmpfile.create_dataset('triggers', data=[])
        tmpfile.create_group('additional_group')
        tmpfile.create_dataset('additional_group/new_dataset', data=(0, 1, 2))
        tmpfile.create_dataset('additional_group/duplicate', data=(3, 4, 5))
        tmpfile.create_dataset('some_other_group/duplicate', data=(6, 7, 8))
        tmpfile.create_group('metadata_group')
        tmpfile.create_dataset('metadata_group/str', data=[])
        tmpfile.create_dataset('metadata_group/float', data=(0,))
        all_locations = h5base._generate_location_names(
            tmpfile, h5base._dataset_locations()
        )
        all_locations.pop('metadata_group')
        all_locations['metadata_group_str'] = '/metadata_group/str'
        all_locations['metadata_group_float'] = '/metadata_group/float'
        existing = h5base._get_bool_dict(tmpfile, all_locations)
        assert not existing['file_meta']
        assert existing['indices']
        assert not existing['waveforms']
        assert not existing['triggers']
        assert not existing['antennas']
        assert not existing['particles_meta']
        assert not existing['antennas_meta']
        assert not existing['rays_meta']
        assert not existing['mc_triggers']
        assert not existing['noise']
        assert existing['new_dataset']
        assert existing['additional_group_duplicate']
        assert existing['some_other_group_duplicate']
        assert not existing['metadata_group_str']
        assert existing['metadata_group_float']

    def test_get_keys_dict(self, h5base, tmpdir):
        """Test that the keys dict for a group is correct"""
        metadata = {
            "color": ['Red', 'Blue', 'Green'],
            "number": [1, 2, 3],
            "zero": [0, 0, 0],
        }
        tmpfile = h5py.File(str(tmpdir.join('tmpfile.h5')))
        tmpfile.create_group('metadata_group')
        str_dataset = tmpfile.create_dataset(
            'metadata_group/str',
            data=[[str.encode(x)] for x in metadata['color']]
        )
        str_dataset.attrs['keys'] = [str.encode('color')]
        float_dataset = tmpfile.create_dataset(
            'metadata_group/float',
            data=np.array([metadata['number'], metadata['zero']]).T
        )
        float_dataset.attrs['keys'] = [str.encode('number'), str.encode('zero')]
        assert h5base._get_keys_dict(tmpfile, 'metadata_group') == {}
        str_keys = h5base._get_keys_dict(tmpfile, 'metadata_group/str')
        assert str_keys['color'] == 0
        float_keys = h5base._get_keys_dict(tmpfile, 'metadata_group/float')
        assert float_keys['number'] == 0
        assert float_keys['zero'] == 1

    def test_convert_ray_value(self, h5base):
        """Check that _convert_ray_value acts as expected"""
        assert h5base._convert_ray_value('direct') == 0
        assert h5base._convert_ray_value('reflected') == 1
        with pytest.raises(ValueError):
            assert h5base._convert_ray_value('hello world')
        assert h5base._convert_ray_value(0) == 0
        assert h5base._convert_ray_value(1) == 1
        assert h5base._convert_ray_value(0.0) == 0
        assert h5base._convert_ray_value(1.0) == 1



@pytest.fixture
def h5writer(tmpdir):
    """Fixture for forming basic HDF5Writer object"""
    return HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w')

class TestHDF5Writer:
    """Tests for HDF5Writer class"""
    def test_creation(self, h5writer, tmpdir):
        """Test initialization of HDF5Writer object"""
        assert h5writer._file_version_major == 1
        assert h5writer._file_version_minor == 0
        assert h5writer.filename == str(tmpdir.join('test_output.h5'))

    def test_open_close(self, h5writer):
        """Test opening and closing the HDF5Writer object"""
        assert not h5writer.is_open
        h5writer.open()
        assert h5writer.is_open
        h5writer.close()
        assert not h5writer.is_open

    def test_context_manager(self, tmpdir):
        """Test opening and closing the HDF5Writer object with a context
        manager"""
        with HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w') as h5writer:
            assert h5writer.is_open
        assert not h5writer.is_open

    def test_getitem(self, h5writer):
        """Test the __getitem__ method for getting groups and datasets"""
        with pytest.raises(IOError):
            h5writer['/event_indices']
        h5writer.open()
        assert h5writer['/event_indices'].name == '/event_indices'
        assert h5writer['event_indices'].name == '/event_indices'
        assert h5writer['indices'].name == '/event_indices'
        assert h5writer['file_meta'].name == '/file_metadata'
        h5writer._file.create_group('/misc')
        h5writer._file.create_dataset('/misc/dataset', [0, 1, 2, 3])
        assert h5writer['misc/dataset'].name == '/misc/dataset'
        assert h5writer['misc']['dataset'].name == '/misc/dataset'
        with pytest.raises(ValueError):
            h5writer['dataset']
        with pytest.raises(ValueError):
            h5writer['nonexistent']
        with pytest.raises(ValueError):
            h5writer[0]
        h5writer.close()

    def test_contains(self, h5writer):
        """Test the __contains__ method for checking existence of datasets"""
        with pytest.raises(IOError):
            '/event_indices' in h5writer
        h5writer.open()
        assert '/event_indices' in h5writer
        assert 'event_indices' in h5writer
        assert 'indices' in h5writer
        assert 'file_meta' in h5writer
        h5writer._file.create_group('/misc')
        h5writer._file.create_dataset('/misc/dataset', [0, 1, 2, 3])
        assert 'misc/dataset' in h5writer
        assert 'dataset' not in h5writer
        assert 'nonexistent' not in h5writer
        h5writer.close()

    def test_delitem(self, h5writer):
        """Test the __delitem__ method for removing groups and datasets"""
        with pytest.raises(IOError):
            del h5writer['/event_indices']
        h5writer.open()
        assert '/event_indices' in h5writer
        del h5writer['/event_indices']
        assert '/event_indices' not in h5writer
        assert '/file_metadata' in h5writer
        assert 'file_meta' in h5writer
        del h5writer['file_meta']
        assert '/file_metadata' not in h5writer
        h5writer.close()

    # def test_write_indices(self, h5writer):
    #     """Test the _write_indices method of HDF5Writer"""
    #     h5writer.open()
    #     assert h5writer['indices'].shape == (0, 0, 2)
    #     h5writer._file.create_dataset('dataset', [0, 1, 2, 3])
    #     h5writer._write_indices('dataset', 0)
    #     assert h5writer['indices'].shape == (1, 1, 2)
    #     assert np.array_equal(h5writer['indices'][:], [[[0, 1]]])
    #     assert np.array_equal(h5writer['indices'].attrs['keys'],
    #                           [str.encode('dataset')])
    #     h5writer._write_indices('dataset', 1, length=2)
    #     assert h5writer['indices'].shape == (1, 1, 2)
    #     assert np.array_equal(h5writer['indices'][:], [[[0, 1], [1, 2]]])
    #     assert np.array_equal(h5writer['indices'].attrs['keys'],
    #                           [str.encode('dataset')])

    def test_set_detector(self, h5writer):
        """Test assigning a detector to the HDF5Writer object"""
        detector = [Antenna(position=(0, 0, -100), noisy=False),
                    Antenna(position=(0, 0, -200), noisy=False)]
        with pytest.raises(IOError):
            h5writer.set_detector(detector)
        h5writer.open()
        assert not h5writer.has_detector
        assert 'antennas' not in h5writer
        assert 'antennas_meta' not in h5writer
        h5writer.set_detector(detector)
        assert h5writer.has_detector
        assert 'antennas' in h5writer
        assert h5writer['antennas'].shape[0] == 2
        assert 'antennas_meta' in h5writer
        assert h5writer['antennas_meta']['str'].shape[0] == 2
        assert h5writer['antennas_meta']['float'].shape[0] == 2
        h5writer.close()

    def test_write_event(self, tmpdir):
        """Test writing event data to the hdf5 file using the add method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=True, write_triggers=False,
                              write_antenna_triggers=False, write_rays=False,
                              write_noise=False, write_waveforms=False,
                              require_trigger=False)
        event = Event(Particle(particle_id=Particle.Type.electron_neutrino,
                               vertex=[100, 200, -500], direction=[0, 0, 1],
                               energy=1e9))
        with pytest.raises(IOError):
            h5writer.add(event)
        h5writer.open()
        assert 'particles_meta' not in h5writer
        h5writer.add(event)
        assert 'particles_meta' in h5writer
        assert h5writer['particles_meta']['str'].shape[0] == 1
        assert h5writer['particles_meta']['float'].shape[0] == 1
        assert h5writer['particles_meta'].attrs['total_thrown'] == 1
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/particles')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]]])
        event2 = Event(Particle(particle_id=Particle.Type.electron_antineutrino,
                                vertex=[0, 0, -100], direction=[0, 0, -1],
                                energy=1e9))
        h5writer.add(event2, events_thrown=5)
        assert h5writer['particles_meta']['str'].shape[0] == 2
        assert h5writer['particles_meta']['float'].shape[0] == 2
        assert h5writer['particles_meta'].attrs['total_thrown'] == 6
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/particles')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]], [[1, 1]]])
        h5writer.close()

    def test_write_global_trigger(self, tmpdir):
        """Test writing global trigger data to the hdf5 file using the add
        method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=False, write_triggers=True,
                              write_antenna_triggers=False, write_rays=False,
                              write_noise=False, write_waveforms=False,
                              require_trigger=False)
        with pytest.raises(IOError):
            h5writer.add(event=None, triggered=True)
        h5writer.open()
        assert 'triggers' not in h5writer
        h5writer.add(event=None, triggered=True)
        assert 'triggers' in h5writer
        assert h5writer['triggers'].shape[0] == 1
        assert np.array_equal(h5writer['triggers'][:], [True])
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/triggers')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]]])
        h5writer.add(event=None, triggered=False)
        assert h5writer['triggers'].shape[0] == 2
        assert np.array_equal(h5writer['triggers'][:], [True, False])
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/triggers')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]], [[1, 1]]])
        h5writer.add(event=None, triggered={'global': True})
        assert h5writer['triggers'].shape[0] == 3
        assert np.array_equal(h5writer['triggers'][:], [True, False, True])
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/triggers')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]], [[1, 1]], [[2, 1]]])
        h5writer.close()

    def test_write_other_triggers(self, tmpdir):
        """Test writing other trigger data (including antenna trigger data) to
        the hdf5 file using the add method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=False, write_triggers=True,
                              write_antenna_triggers=True, write_rays=False,
                              write_noise=False, write_waveforms=False,
                              require_trigger=False)
        with pytest.raises(IOError):
            h5writer.add(event=None, triggered=True)
        h5writer.open()
        # with pytest.raises(ValueError):
        #     h5writer.add(event=None, triggered=True)
        detector = [Antenna(position=(0, 0, -100), noisy=False),
                    Antenna(position=(0, 0, -200), noisy=False)]
        for ant in detector:
            ant.signals.append(Signal([0, 1e-9], [0, 1]))
        h5writer.set_detector(detector)
        assert 'mc_triggers' not in h5writer
        h5writer.add(event=None, triggered=True)
        assert 'mc_triggers' in h5writer
        assert h5writer['mc_triggers'].shape[0] == 1
        assert h5writer['mc_triggers'].shape[1] == len(detector)
        assert np.array_equal(h5writer['mc_triggers'].attrs['keys'],
                              [str.encode('antenna_0'),
                               str.encode('antenna_1')])
        assert np.array_equal(h5writer['mc_triggers'][:], [[True, True]])
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/triggers'),
                               str.encode('/monte_carlo_data/triggers')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1], [0, 1]]])
        detector[0].signals.append(Signal([2e-9, 3e-9], [1, 0]))
        h5writer.add(event=None, triggered={'global': False, 'extra': True})
        assert h5writer['mc_triggers'].shape[0] == 3
        assert h5writer['mc_triggers'].shape[1] == len(detector)+1
        assert np.array_equal(h5writer['mc_triggers'].attrs['keys'],
                              [str.encode('antenna_0'),
                               str.encode('antenna_1'),
                               str.encode('extra')])
        assert np.array_equal(h5writer['mc_triggers'][:],
                              [[True, True, False], [True, True, True],
                               [True, False, True]])
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/triggers'),
                               str.encode('/monte_carlo_data/triggers')])
        assert np.array_equal(h5writer['indices'][:],
                              [[[0, 1], [0, 1]], [[1, 1], [1, 2]]])
        h5writer.close()

    def test_write_rays(self, tmpdir):
        """Test writing ray data to the hdf5 file using the add method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=False, write_triggers=False,
                              write_antenna_triggers=False, write_rays=True,
                              write_noise=False, write_waveforms=False,
                              require_trigger=False)
        detector = [Antenna(position=(0, 0, -100), noisy=False),
                    Antenna(position=(0, 0, -200), noisy=False)]
        rays = [RayTracer((100, 200, -500), ant.position).solutions
                for ant in detector]
        polarizations = [[(0, 0, 1), (0, 0, -1)] for _ in detector]
        with pytest.raises(IOError):
            h5writer.add(event=None, ray_paths=rays, polarizations=polarizations)
        h5writer.open()
        with pytest.raises(ValueError):
            h5writer.add(event=None, ray_paths=rays, polarizations=polarizations)
        h5writer.set_detector(detector)
        assert 'rays_meta' not in h5writer
        h5writer.add(event=None, ray_paths=rays, polarizations=polarizations)
        assert 'rays_meta' in h5writer
        assert h5writer['rays_meta']['float'].shape[0] == 2
        assert h5writer['rays_meta']['float'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/rays')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 2]]])
        h5writer.add(event=None, ray_paths=rays, polarizations=polarizations)
        assert h5writer['rays_meta']['float'].shape[0] == 4
        assert h5writer['rays_meta']['float'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/rays')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 2]], [[2, 2]]])
        h5writer.close()

    def test_write_noise(self, tmpdir):
        """Test writing noise basis data to the hdf5 file using the add method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=False, write_triggers=False,
                              write_antenna_triggers=False, write_rays=False,
                              write_noise=True, write_waveforms=False,
                              require_trigger=False)
        detector = [Antenna(position=(0, 0, -100), noisy=True, temperature=300,
                            resistance=100, freq_range=[500e6, 750e6]),
                    Antenna(position=(0, 0, -200), noisy=True, temperature=300,
                            resistance=100, freq_range=[500e6, 750e6])]
        for ant in detector:
            ant.signals.append(Signal([0, 1e-9, 2e-9], [0, 1, 2]))
            ant.all_waveforms
        with pytest.raises(IOError):
            h5writer.add(event=None)
        h5writer.open()
        with pytest.raises(ValueError):
            h5writer.add(event=None)
        h5writer.set_detector(detector)
        assert 'noise' not in h5writer
        h5writer.add(event=None)
        assert 'noise' in h5writer
        assert h5writer['noise'].shape[0] == 1
        assert h5writer['noise'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/noise')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]]])
        h5writer.add(event=None)
        assert h5writer['noise'].shape[0] == 2
        assert h5writer['noise'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/noise')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]], [[1, 1]]])
        h5writer.close()

    def test_write_waveforms(self, tmpdir):
        """Test writing waveform data to the hdf5 file using the add method"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=False, write_triggers=False,
                              write_antenna_triggers=False, write_rays=False,
                              write_noise=False, write_waveforms=True,
                              require_trigger=False)
        detector = [Antenna(position=(0, 0, -100), noisy=False),
                    Antenna(position=(0, 0, -200), noisy=False)]
        for ant in detector:
            ant.signals.append(Signal([0, 1e-9, 2e-9], [0, 1, 2]))
        with pytest.raises(IOError):
            h5writer.add(event=None)
        h5writer.open()
        with pytest.raises(ValueError):
            h5writer.add(event=None)
        h5writer.set_detector(detector)
        assert 'waveforms' not in h5writer
        h5writer.add(event=None)
        assert 'waveforms' in h5writer
        assert h5writer['waveforms'].shape[0] == 1
        assert h5writer['waveforms'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/waveforms')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]]])
        detector[0].signals.append(Signal([2e-9, 3e-9], [1, 0]))
        h5writer.add(event=None)
        assert 'waveforms' in h5writer
        assert h5writer['waveforms'].shape[0] == 3
        assert h5writer['waveforms'].shape[1] == len(detector)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/data/waveforms')])
        assert np.array_equal(h5writer['indices'][:], [[[0, 1]], [[1, 2]]])
        h5writer.close()

    def test_require_trigger(self, tmpdir):
        """Test that the require_trigger argument is obeyed when adding data"""
        h5writer = HDF5Writer(str(tmpdir.join('test_output.h5')), mode='w',
                              write_particles=True, write_triggers=True,
                              write_antenna_triggers=True, write_rays=True,
                              write_noise=False, write_waveforms=True,
                              require_trigger=True)
        event = Event(Particle(particle_id=Particle.Type.electron_neutrino,
                               vertex=[100, 200, -500], direction=[0, 0, 1],
                               energy=1e9))
        detector = [Antenna(position=(0, 0, -100), noisy=False),
                    Antenna(position=(0, 0, -200), noisy=False)]
        rays = [RayTracer(event.roots[0].vertex, ant.position).solutions
                for ant in detector]
        polarizations = [[(0, 0, 1), (0, 0, -1)] for _ in detector]
        for ant in detector:
            ant.signals.append(Signal([0, 1e-9, 2e-9], [0, 1, 2]))
        h5writer.open()
        h5writer.set_detector(detector)
        h5writer.add(event=event, triggered=False, ray_paths=rays,
                     polarizations=polarizations)
        assert 'indices' in h5writer
        assert 'particles_meta' in h5writer
        assert 'triggers' in h5writer
        assert 'mc_triggers' in h5writer
        assert 'rays_meta' not in h5writer
        assert 'waveforms' not in h5writer
        h5writer.add(event=event, triggered=True, ray_paths=rays,
                     polarizations=polarizations)
        assert 'rays_meta' in h5writer
        assert 'waveforms' in h5writer
        h5writer.close()

    def test_create_analysis_group(self, h5writer):
        """Test the creation of an analysis group in the hdf5 file"""
        with pytest.raises(IOError):
            h5writer.create_analysis_group('new_group')
        h5writer.open()
        assert 'new_group' not in h5writer
        h5writer.create_analysis_group('new_group')
        assert 'new_group' in h5writer
        assert (h5writer['new_group'].name ==
                '/monte_carlo_data/analysis/new_group')
        h5writer.close()

    def test_create_analysis_dataset(self, h5writer):
        """Test the creation of an analysis dataset in the hdf5 file"""
        with pytest.raises(IOError):
            h5writer.create_analysis_dataset('new_dataset', data=[0, 1, 2])
        h5writer.open()
        assert 'new_dataset' not in h5writer
        h5writer.create_analysis_dataset('new_dataset', data=[0, 1, 2])
        assert 'new_dataset' in h5writer
        assert (h5writer['new_dataset'].name ==
                '/monte_carlo_data/analysis/new_dataset')
        h5writer.close()

    def test_create_analysis_metadataset(self, h5writer):
        """Test the creation of an analysis metadata group in the hdf5 file"""
        with pytest.raises(IOError):
            h5writer.create_analysis_metadataset('new_metadata')
        h5writer.open()
        assert 'new_metadata' not in h5writer
        h5writer.create_analysis_metadataset('new_metadata')
        assert 'new_metadata' in h5writer
        assert (h5writer['new_metadata'].name ==
                '/monte_carlo_data/analysis/new_metadata')
        assert (h5writer['new_metadata']['str'].name ==
                '/monte_carlo_data/analysis/new_metadata/str')
        assert (h5writer['new_metadata']['float'].name ==
                '/monte_carlo_data/analysis/new_metadata/float')
        h5writer.close()

    def test_add_analysis_metadata(self, h5writer):
        """Test writing data to an analysis metadata group in the hdf5 file"""
        h5writer.open()
        h5writer.create_analysis_metadataset('new_metadata')
        h5writer.add_analysis_metadata('new_metadata',
                                       {'color': 'red', 'number': 1, 'zero': 0})
        assert np.array_equal(h5writer['new_metadata']['str'][:], ['red'])
        assert np.array_equal(h5writer['new_metadata']['str'].attrs['keys'],
                              [str.encode('color')])
        assert np.array_equal(h5writer['new_metadata']['float'][:], [1, 0])
        assert np.array_equal(h5writer['new_metadata']['float'].attrs['keys'],
                              [str.encode('number'), str.encode('zero')])
        h5writer.close()

    def test_add_analysis_indices(self, h5writer):
        """Test writing analysis dataset indices in the hdf5 file"""
        h5writer.open()
        h5writer.create_analysis_dataset('new_dataset',
                                         data=[[[0, 1]], [[2, 3]], [[4, 5]]])
        h5writer.add_analysis_indices('new_dataset', 0, 0, 1)
        h5writer.add_analysis_indices('new_dataset', 1, 1, 2)
        assert np.array_equal(h5writer['indices'].attrs['keys'],
                              [str.encode('/monte_carlo_data/analysis/new_dataset')])
        assert np.array_equal(h5writer['indices'][:],
                              [[[0, 1]], [[1, 2]]])
        h5writer.close()

    def test_add_file_metadata(self, h5writer):
        """Test adding file metadata to the hdf5 file"""
        with pytest.raises(IOError):
            h5writer.add_file_metadata({'color': 'red', 'number': 1, 'zero': 0})
        h5writer.open()
        str_key_idx = len(h5writer['file_meta']['str'].attrs['keys'])
        float_key_idx = len(h5writer['file_meta']['float'].attrs['keys'])
        h5writer.add_file_metadata({'color': 'red', 'number': 1, 'zero': 0})
        assert (h5writer['file_meta']['str'].attrs['keys'][str_key_idx] ==
                str.encode('color'))
        assert h5writer['file_meta']['str'][str_key_idx] == 'red'
        assert (h5writer['file_meta']['float'].attrs['keys'][float_key_idx] ==
                str.encode('number'))
        assert h5writer['file_meta']['float'][float_key_idx] == 1
        assert (h5writer['file_meta']['float'].attrs['keys'][float_key_idx+1] ==
                str.encode('zero'))
        assert h5writer['file_meta']['float'][float_key_idx+1] == 0
        h5writer.close()
