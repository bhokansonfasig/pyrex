"""File containing tests of pyrex detector module"""

import pytest

from config import SEED

from pyrex.detector import AntennaSystem, Detector, CombinedDetector
from pyrex.signals import Signal
from pyrex.antenna import Antenna

import numpy as np



@pytest.fixture
def ant_sys():
    """Fixture for forming basic AntennaSystem object from class"""
    return AntennaSystem(Antenna)

@pytest.fixture
def ant_obj_sys():
    """Fixture for forming basic AntennaSystem object from Antenna object"""
    ant = Antenna(position=[0,0,-250], temperature=300,
                  freq_range=[500e6, 750e6], resistance=100)
    return AntennaSystem(ant)

@pytest.fixture
def halver():
    """Fixture for forming AntennaSystem which halves signals"""
    ant = Antenna(position=[0,0,-250], temperature=300,
                  freq_range=[500e6, 750e6], resistance=100)
    ant_sys = AntennaSystem(ant)
    ant_sys.front_end = lambda signal: Signal(signal.times, signal.values/2,
                                              value_type=signal.value_type)
    ant_sys.trigger = lambda signal: np.max(signal.values)>1
    return ant_sys

@pytest.fixture
def noiseless_halver():
    """Fixture for forming AntennaSystem which halves signals"""
    ant = Antenna(position=[0,0,-250], noisy=False)
    ant_sys = AntennaSystem(ant)
    ant_sys.front_end = lambda signal: Signal(signal.times, signal.values/2,
                                              value_type=signal.value_type)
    ant_sys.trigger = lambda signal: np.max(signal.values)>1
    return ant_sys


class TestAntennaSystem:
    """Tests for AntennaSystem class"""
    def test_creation(self, ant_sys, ant_obj_sys):
        """Test initialization of antenna"""
        assert ant_sys._antenna_class == Antenna
        assert ant_obj_sys._antenna_class == Antenna
        assert isinstance(ant_obj_sys.antenna, Antenna)

    def test_setup_antenna(self, ant_sys, ant_obj_sys):
        """Test that the setup_antenna method prepares the antenna correctly"""
        assert not hasattr(ant_sys, "antenna") or ant_sys.antenna is None
        ant_sys.setup_antenna([0, 0, -250], temperature=300,
                              freq_range=[500e6, 750e6], resistance=100)
        assert np.array_equal(ant_sys.antenna.position, [0,0,-250])
        assert np.array_equal(ant_sys.antenna.z_axis, [0,0,1])
        assert np.array_equal(ant_sys.antenna.x_axis, [1,0,0])
        assert ant_sys.antenna.antenna_factor == 1
        assert ant_sys.antenna.efficiency == 1
        assert np.array_equal(ant_sys.antenna.freq_range, [500e6, 750e6])
        assert ant_sys.antenna.temperature == 300
        assert ant_sys.antenna.resistance == 100
        assert ant_sys.antenna.noisy
        assert np.array_equal(ant_obj_sys.antenna.position, [0,0,-250])
        assert np.array_equal(ant_obj_sys.antenna.z_axis, [0,0,1])
        assert np.array_equal(ant_obj_sys.antenna.x_axis, [1,0,0])
        assert ant_obj_sys.antenna.antenna_factor == 1
        assert ant_obj_sys.antenna.efficiency == 1
        assert np.array_equal(ant_obj_sys.antenna.freq_range, [500e6, 750e6])
        assert ant_obj_sys.antenna.temperature == 300
        assert ant_obj_sys.antenna.resistance == 100
        assert ant_obj_sys.antenna.noisy

    def test_default_frontend(self, ant_sys):
        """Test that the default front end just passes along the signal"""
        signal = Signal([0,1,2], [1,2,1])
        fe_sig = ant_sys.front_end(signal)
        assert np.array_equal(fe_sig.times, signal.times)
        assert np.array_equal(fe_sig.values, signal.values)
        assert fe_sig.value_type == signal.value_type

    def test_is_hit(self, noiseless_halver):
        """Test that is_hit works with the front end"""
        assert not noiseless_halver.is_hit
        assert not noiseless_halver.antenna.is_hit
        noiseless_halver.antenna.signals.append(Signal([0, 1e-9], [1, 1]))
        assert not noiseless_halver.is_hit
        assert noiseless_halver.antenna.is_hit
        noiseless_halver.antenna.signals.append(Signal([0, 1e-9], [4, 4]))
        assert noiseless_halver.is_hit
        assert noiseless_halver.antenna.is_hit

    def test_is_hit_mc_truth(self, halver):
        """Test that is_hit_mc_truth works with the front end"""
        np.random.seed(SEED)
        halver.antenna.noise_rms = 100
        assert not halver.is_hit_mc_truth
        assert not halver.antenna.is_hit_mc_truth
        halver.antenna.signals.append(Signal([0, 1e-9], [1, 1]))
        assert not halver.is_hit_mc_truth
        assert not halver.antenna.is_hit_mc_truth
        halver.antenna.signals.append(Signal([0, 1e-9], [4, 4]))
        assert not halver.is_hit_mc_truth
        assert not halver.antenna.is_hit_mc_truth

    def test_is_hit_during(self, noiseless_halver):
        """Test that is_hit_during works with the front end"""
        noiseless_halver.antenna.signals.append(Signal([0, 1e-9], [0, 1]))
        assert not noiseless_halver.is_hit_during([0, 1, 2])
        noiseless_halver.antenna.signals.append(Signal([0, 1e-9], [0, 4]))
        assert noiseless_halver.is_hit_during([0, 1e-9, 2e-9])
        assert not noiseless_halver.is_hit_during([-2e-9, -1e-9, 0])
        assert not noiseless_halver.is_hit_during([2e-9, 3e-9, 4e-9])

    def test_signals(self, halver):
        """Test that front end is applied to signals array"""
        halver.antenna.signals.append(Signal([0, 1e-9], [2, 2]))
        assert halver.signals != []
        assert np.array_equal(halver.signals[0].values,
                              halver.antenna.signals[0].values/2)

    def test_waveforms(self, halver):
        """Test that front end is applied to waveforms array"""
        halver.antenna.signals.append(Signal([0, 0.5e-9, 1e-9], [2, 4, 2]))
        assert halver.waveforms != []
        assert np.array_equal(halver.waveforms[0].values,
                              halver.antenna.waveforms[0].values/2)

    def test_all_waveforms(self, halver):
        """Test that front end is applied to all_waveforms array"""
        halver.antenna.signals.append(Signal([0, 0.5e-9, 1e-9], [0.2, 0.4, 0.2]))
        assert halver.waveforms == []
        assert np.array_equal(halver.all_waveforms[0].values,
                              halver.antenna.all_waveforms[0].values/2)

    def test_full_waveform(self, halver):
        """Test that front end is applied to full waveform"""
        halver.antenna.signals.append(Signal([0, 0.5e-9, 1e-9], [2, 4, 2]))
        assert np.array_equal(halver.full_waveform([-0.5e-9, 0, 0.5e-9, 1e-9, 1.5e-9]).values,
                              halver.antenna.full_waveform([-0.5e-9, 0, 0.5e-9, 1e-9, 1.5e-9]).values/2)

    def test_receive(self, ant_obj_sys):
        """Test that receive is passed along to underlying antenna"""
        assert ant_obj_sys.signals == []
        assert ant_obj_sys.antenna.signals == []
        ant_obj_sys.receive(Signal([0,1,2], [1,2,1], Signal.Type.voltage))
        assert ant_obj_sys.signals != []
        assert ant_obj_sys.antenna.signals != []

    def test_clear(self, ant_obj_sys):
        """Test that clear clears the system and the underlying antenna"""
        ant_obj_sys.receive(Signal([0,1,2], [1,2,1], Signal.Type.voltage))
        assert ant_obj_sys.signals != []
        assert ant_obj_sys.antenna.signals != []
        ant_obj_sys.clear()
        assert ant_obj_sys.signals == []
        assert ant_obj_sys.antenna.signals == []

    def test_trigger(self, ant_obj_sys):
        """Test that the default trigger falls back on the antenna's trigger"""
        assert ant_obj_sys.trigger(Signal([0], [0])) == True
        ant_obj_sys.antenna.trigger = lambda signal: False
        assert ant_obj_sys.trigger(Signal([0], [0])) == False



class DummyString(Detector):
    def set_positions(self, x, y, n_antennas):
        for i in range(n_antennas):
            self.antenna_positions.append((x, y, -i))

class DummyDetector(Detector):
    def set_positions(self, xy_pos):
        for pos in xy_pos:
            self.subsets.append(DummyString(pos[0], pos[1], 2))

@pytest.fixture
def dummy_str():
    """Fixture for a DummyString object"""
    return DummyString(0, 0, 5)

@pytest.fixture
def dummy_det():
    """Fixture for a DummyDetector object"""
    return DummyDetector([(0,0), (1,0), (0,1), (1,1)])

class TestDetector:
    """Tests for Detector class"""
    def test_creation_failure(self):
        """Test that initialization fails for plain base Detector class"""
        with pytest.raises(NotImplementedError):
            det = Detector()

    def test_creation(self, dummy_str, dummy_det):
        """Test initialization of Detector subclasses"""
        assert np.array_equal(dummy_str.antenna_positions, 
                              [(0,0,0), (0,0,-1), (0,0,-2), (0,0,-3), (0,0,-4)])
        assert dummy_str.subsets == []
        assert np.array_equal(dummy_det.antenna_positions,
                              [[(0,0,0), (0,0,-1)], [(1,0,0), (1,0,-1)],
                               [(0,1,0), (0,1,-1)], [(1,1,0), (1,1,-1)]])
        assert len(dummy_det.subsets) == 4
        assert isinstance(dummy_det.subsets[0], DummyString)

    def test_addition(self, dummy_str, dummy_det, ant_obj_sys):
        """Test addition of antennas or detectors to Detector objects"""
        combined = dummy_det + dummy_str
        assert isinstance(combined, CombinedDetector)
        assert combined.subsets == [dummy_det] + [dummy_str]
        assert combined.antenna_positions == [dummy_det.antenna_positions,
                                              dummy_str.antenna_positions]
        ant_obj_sys.position = ant_obj_sys.antenna.position
        combined2 = dummy_det + ant_obj_sys
        assert isinstance(combined2, CombinedDetector)
        assert combined2.subsets == [dummy_det] + [ant_obj_sys]
        assert combined2.antenna_positions == [dummy_det.antenna_positions,
                                               ant_obj_sys.position]

    def test_build_antennas(self, dummy_str, dummy_det):
        """Test build_antennas for Detector subclassed objects with and without
        further subsets"""
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        assert len(dummy_str.subsets) == 5
        for i in range(5):
            assert isinstance(dummy_str.subsets[i], Antenna)
            assert not dummy_str.subsets[i].noisy
        dummy_det.build_antennas(antenna_class=Antenna, noisy=False)
        assert len(dummy_det.subsets) == 4
        for i in range(4):
            assert isinstance(dummy_det.subsets[i], DummyString)
            assert len(dummy_det.subsets[i]) == 2
            for j in range(2):
                assert isinstance(dummy_det.subsets[i][j], Antenna)
                assert not dummy_det.subsets[i][j].noisy

    def test_detector_len(self, dummy_str, dummy_det):
        """Test that the length of the detector object is the total number of
        antennas"""
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        assert len(dummy_str) == 5
        dummy_det.build_antennas(antenna_class=Antenna, noisy=False)
        assert len(dummy_det) == 4*2

    def test_iteration(self, dummy_str, dummy_det):
        """Test that the detector object can be directly iterated to iterate
        over all antennas"""
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        for ant in dummy_str:
            assert isinstance(ant, Antenna)
        dummy_det.build_antennas(antenna_class=Antenna, noisy=False)
        for ant in dummy_det:
            assert isinstance(ant, Antenna)

    def test_detector_indexing(self, dummy_str, dummy_det):
        """Test that the detector object can be indexed"""
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        for i, ant in enumerate(dummy_str):
            assert dummy_str[i] == ant
        dummy_det.build_antennas(antenna_class=Antenna, noisy=False)
        for i, ant in enumerate(dummy_det):
            assert dummy_det[i] == ant

    def test_triggered(self, dummy_str):
        """Test that the default trigger just checks trigger of any antenna"""
        assert not dummy_str.triggered()
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        assert not dummy_str.triggered()
        dummy_str.subsets[0].signals.append(Signal([0,1e-9], [0,1]))
        assert dummy_str.triggered()

    def test_clear(self, dummy_str):
        """Test that the clear method clears all antennas"""
        dummy_str.build_antennas(antenna_class=Antenna, noisy=False)
        for i in range(5):
            assert dummy_str.subsets[i].signals == []
            dummy_str.subsets[i].signals.append(Signal([0], [0]))
            assert dummy_str.subsets[i].signals != []
        dummy_str.clear()
        for i in range(5):
            assert dummy_str.subsets[i].signals == []


@pytest.fixture
def dummy_combo():
    """Fixture for a DummyDetector object"""
    str1 = DummyString(0, -1, 5)
    str2 = DummyString(0, 1, 5)
    return CombinedDetector(str1, str2)

class TestCombinedDetector:
    """Tests for Detector class"""
    def test_creation(self, dummy_combo):
        """Test that initialization of CombinedDetector class"""
        assert len(dummy_combo.subsets) == 2
        for string in dummy_combo.subsets:
            assert isinstance(string, DummyString)
        assert dummy_combo.antenna_positions == [string.antenna_positions
                                                 for string in dummy_combo.subsets]
    
    def test_addition(self, dummy_combo, dummy_str, ant_obj_sys):
        """Test addition of antennas or detectors to Detector objects"""
        combined = dummy_combo + dummy_str
        assert isinstance(combined, CombinedDetector)
        assert combined.subsets == dummy_combo.subsets + [dummy_str]
        assert combined.antenna_positions == (dummy_combo.antenna_positions +
                                              [dummy_str.antenna_positions])
        ant_obj_sys.position = ant_obj_sys.antenna.position
        combined2 = dummy_combo + ant_obj_sys
        assert isinstance(combined2, CombinedDetector)
        assert combined2.subsets == dummy_combo.subsets + [ant_obj_sys]
        assert combined2.antenna_positions == (dummy_combo.antenna_positions +
                                               [ant_obj_sys.position])
        combined3 = CombinedDetector(DummyString(0, -1, 5), DummyString(0, 1, 5))
        combined3 += dummy_str
        assert isinstance(combined, CombinedDetector)
        assert combined.subsets == dummy_combo.subsets + [dummy_str]
        assert combined.antenna_positions == (dummy_combo.antenna_positions +
                                              [dummy_str.antenna_positions])

    def test_triggered(self, dummy_combo):
        """Test that CombinedDetector trigger just checks trigger of any subset"""
        assert not dummy_combo.triggered()
        dummy_combo.build_antennas(antenna_class=Antenna, noisy=False)
        assert not dummy_combo.triggered()
        dummy_combo.subsets[0][0].signals.append(Signal([0,1e-9], [0,1]))
        assert dummy_combo.triggered()
