"""File containing tests of pyrex digsig module"""

import pytest

from pyrex.antenna import Antenna
from pyrex.signals import Signal

import numpy as np



@pytest.fixture
def antenna():
    """Fixture for forming basic Antenna object"""
    return Antenna(position=[0,0,-250], temperature=300,
                   freq_range=[500e6, 750e6], resistance=1000)

@pytest.fixture
def dipole():
    """Fixture for forming basic DipoleAntenna object"""
    return Antenna(name="ant", position=[0,0,-250], center_frequency=250,
                   bandwidth=100, resistance=1000, effective_height=1.0,
                   threshold=5E-6)


class TestAntenna:
    """Tests for Antenna class"""
    def test_creation(self, antenna):
        """Test that the antenna's creation goes as expected"""
        assert np.array_equal(antenna.position, [0,0,-250])
        assert antenna.temperature == 300
        assert np.array_equal(antenna.freq_range, [500e6, 750e6])
        assert antenna.resistance == 1000
        assert antenna.noisy

    def test_is_hit(self, antenna):
        """Test that is_hit is true when there is a signal and false otherwise"""
        assert not(antenna.is_hit)
        antenna.signals.append(Signal([0],[0]))
        assert antenna.is_hit

    def test_is_hit_not_writable(self, antenna):
        """Test that is_hit cannot be assigned to"""
        with pytest.raises(AttributeError):
            antenna.is_hit = True

    def test_clear(self, antenna):
        """Test that clear emptys signals list"""
        antenna.signals.append(Signal([0],[0]))
        antenna.clear()
        assert antenna.signals == []

    def test_no_waveforms(self, antenna):
        """Test that waveforms returns an empty list if there are no signals"""
        assert antenna.waveforms == []

    def test_waveforms_exist(self, antenna):
        """Test that waveforms returns a waveform when there is a signal"""
        antenna.signals.append(Signal([0],[0]))
        assert antenna.waveforms != []
        assert isinstance(antenna.waveforms[0], Signal)

    def test_noises_not_recalculated(self, antenna):
        """Test that noise signals aren't recalculated every time"""
        antenna.signals.append(Signal([0],[1]))
        waveforms1 = antenna.waveforms
        noises1 = antenna._noises
        waveforms2 = antenna.waveforms
        noises2 = antenna._noises
        assert noises1 == noises2
