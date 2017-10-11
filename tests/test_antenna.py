"""File containing tests of pyrex digsig module"""

import pytest

from pyrex.antenna import Antenna, DipoleAntenna
from pyrex.signals import Signal, ValueTypes

import numpy as np



@pytest.fixture
def antenna():
    """Fixture for forming basic Antenna object"""
    return Antenna(position=[0,0,-250], temperature=300,
                   freq_range=[500e6, 750e6], resistance=1000)

@pytest.fixture
def dipole():
    """Fixture for forming basic DipoleAntenna object"""
    return DipoleAntenna(name="ant", position=[0,0,-250], center_frequency=250e6,
                         bandwidth=100e6, resistance=1000, effective_height=1.0,
                         trigger_threshold=5E-6)


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

    def test_default_trigger(self, antenna):
        """Test that the antenna triggers on empty signal"""
        assert antenna.trigger(Signal([0],[0]))

    def test_default_response(self, antenna):
        """Test that the frequency response is always 1"""
        assert np.array_equal(antenna.response(np.logspace(0,10)), np.ones(50))

    def test_receive(self, antenna):
        """Test that the antenna properly receives signals"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], ValueTypes.voltage))
        assert len(antenna.signals) > 0

    def test_no_waveforms(self, antenna):
        """Test that waveforms returns an empty list if there are no signals"""
        assert antenna.waveforms == []

    def test_waveforms_exist(self, antenna):
        """Test that waveforms returns a waveform when a signal has been received"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], ValueTypes.voltage))
        assert antenna.waveforms != []
        assert isinstance(antenna.waveforms[0], Signal)
        assert antenna._noises != []
        assert antenna._triggers == [True]

    def test_delay_noise_calculation(self, antenna):
        """Test that antenna noise isn't calculated until it is needed"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], ValueTypes.voltage))
        assert antenna._noises == []
        antenna.waveforms
        assert antenna._noises != []

    def test_noises_not_recalculated(self, antenna):
        """Test that noise signals aren't recalculated every time"""
        antenna.signals.append(Signal([0],[1]))
        waveforms1 = antenna.waveforms
        noises1 = antenna._noises
        waveforms2 = antenna.waveforms
        noises2 = antenna._noises
        assert noises1 == noises2

    def test_no_trigger_no_waveform(self, antenna):
        """Test that signals which don't trigger don't appear in waveforms,
        but do appear in all_waveforms"""
        antenna.trigger = lambda signal: False
        antenna.signals.append(Signal([0],[1]))
        assert antenna.is_hit == False
        assert antenna.waveforms == []
        assert antenna.all_waveforms != []



def test_dipole_response(dipole):
    """Test that the response of the dipole antenna is as expected"""
    responses = dipole.response([150e6, 225e6, 250e6, 275e6, 350e6])
    assert np.abs(responses[0]) < .5
    assert np.abs(responses[1]) == pytest.approx(.9, rel=0.1)
    assert np.abs(responses[2]) == pytest.approx(1, rel=0.01)
    assert np.abs(responses[3]) == pytest.approx(.9, rel=0.1)
    assert np.abs(responses[4]) < .5
