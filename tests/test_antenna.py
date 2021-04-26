"""File containing tests of pyrex antenna module"""

import pytest

from config import SEED

from pyrex.antenna import Antenna, DipoleAntenna
from pyrex.signals import Signal
from pyrex.ice_model import ice

import numpy as np
import scipy.constants



@pytest.fixture
def antenna():
    """Fixture for forming basic Antenna object"""
    return Antenna(position=[0,0,-250], temperature=300,
                   freq_range=[500e6, 750e6], resistance=100)

@pytest.fixture
def triggerable_antenna():
    """Fixture for an antenna with a real trigger"""
    ant = Antenna(position=[0,0,-200], noisy=False)
    ant.trigger = lambda signal: np.max(np.abs(signal.values))>1
    return ant


class TestAntenna:
    """Tests for Antenna class"""
    def test_creation(self, antenna):
        """Test initialization of antenna"""
        assert np.array_equal(antenna.position, [0,0,-250])
        assert np.array_equal(antenna.z_axis, [0,0,1])
        assert np.array_equal(antenna.x_axis, [1,0,0])
        assert antenna.antenna_factor == 1
        assert antenna.efficiency == 1
        assert np.array_equal(antenna.freq_range, [500e6, 750e6])
        assert antenna.temperature == 300
        assert antenna.resistance == 100
        assert antenna.noisy

    def test_set_orientation(self, antenna):
        """Test that set_orientation properly sets x and z axes,
        and fails when they aren't orthogonal"""
        antenna.set_orientation(z_axis=[1,0,0], x_axis=[0,1,0])
        assert np.array_equal(antenna.z_axis, [1,0,0])
        assert np.array_equal(antenna.x_axis, [0,1,0])
        antenna.set_orientation()
        assert np.array_equal(antenna.z_axis, [0,0,1])
        assert np.array_equal(antenna.x_axis, [1,0,0])
        antenna.set_orientation(z_axis=[0,0,5])
        assert np.array_equal(antenna.z_axis, [0,0,1])
        assert np.array_equal(antenna.x_axis, [1,0,0])
        with pytest.raises(ValueError):
            antenna.set_orientation(z_axis=[2,0,0], x_axis=[1,1,0])

    def test_is_hit(self, triggerable_antenna):
        """Test that is_hit is true when there is a triggering signal and false otherwise"""
        assert not triggerable_antenna.is_hit
        triggerable_antenna.signals.append(Signal([0, 1], [0, 0]))
        assert not triggerable_antenna.is_hit
        triggerable_antenna.signals.append(Signal([0, 1], [0, 2]))
        assert triggerable_antenna.is_hit

    def test_is_hit_not_writable(self, antenna):
        """Test that is_hit cannot be assigned to"""
        with pytest.raises(AttributeError):
            antenna.is_hit = True

    def test_is_hit_mc_truth(self, triggerable_antenna):
        """Test that is_hit_mc_truth appropriately rejects noise triggers"""
        assert not triggerable_antenna.is_hit_mc_truth
        triggerable_antenna.signals.append(Signal([0, 1], [0, 0]))
        assert not triggerable_antenna.is_hit_mc_truth
        triggerable_antenna.signals.append(Signal([0, 1], [0, 2]))
        assert triggerable_antenna.is_hit_mc_truth
        np.random.seed(SEED)
        noisy_antenna = Antenna(position=[0,0,-200], freq_range=[500e6, 750e6],
                                noise_rms=100)
        assert not noisy_antenna.is_hit_mc_truth
        noisy_antenna.signals.append(Signal([0, 1e-9], [0, 0]))
        assert not noisy_antenna.is_hit_mc_truth
        noisy_antenna.signals.append(Signal([0, 1e-9], [0, 2]))
        assert not noisy_antenna.is_hit_mc_truth

    def test_is_hit_during(self, triggerable_antenna):
        """Test that is_hit_during works as expected"""
        triggerable_antenna.signals.append(Signal([1], [2]))
        assert triggerable_antenna.is_hit_during([0, 1, 2])
        assert not triggerable_antenna.is_hit_during([-2, -1, 0])
        assert not triggerable_antenna.is_hit_during([2, 3, 4])

    def test_clear(self, antenna):
        """Test that clear emptys signals list"""
        antenna.signals.append(Signal([0],[0]))
        assert antenna.signals != []
        antenna.clear()
        assert antenna.signals == []

    def test_default_trigger(self, antenna):
        """Test that the antenna triggers on empty signal"""
        assert antenna.trigger(Signal([0],[0]))

    def test_default_response(self, antenna):
        """Test that the frequency response is always 1"""
        assert np.array_equal(
            antenna.frequency_response(np.logspace(0,10)),
            np.ones(50)
        )

    def test_default_directional_gain(self, antenna):
        """Test that the directional gain is always 1"""
        thetas = np.linspace(0, np.pi, 7)
        phis = np.linspace(0, 2*np.pi, 13)
        gains = []
        for theta in thetas:
            for phi in phis:
                gains.append(antenna.directional_gain(theta, phi))
        assert np.array_equal(gains, np.ones(7*13))

    def test_default_polarization_gain(self, antenna):
        """Test that the polarization gain is always 1"""
        xs = np.linspace(0, 1, 3)
        ys = np.linspace(0, 1, 3)
        zs = np.linspace(0, 1, 3)
        gains = []
        for x in xs:
            for y in ys:
                for z in zs:
                    gains.append(antenna.polarization_gain((x,y,z)))
        assert np.array_equal(gains, np.ones(3**3))

    def test_receive(self, antenna):
        """Test that the antenna properly receives signals"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], Signal.Type.voltage))
        assert len(antenna.signals) > 0

    def test_no_waveforms(self, antenna):
        """Test that waveforms returns an empty list if there are no signals"""
        assert antenna.waveforms == []
        assert antenna.all_waveforms == []

    def test_waveforms_exist(self, antenna):
        """Test that waveforms returns a waveform when a signal has been received"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], Signal.Type.voltage))
        assert antenna.waveforms != []
        assert antenna.all_waveforms != []
        assert isinstance(antenna.waveforms[0], Signal)
        assert antenna._noise_master is not None
        assert antenna._triggers == [True]

    def test_delay_noise_calculation(self, antenna):
        """Test that antenna noise isn't calculated until it is needed"""
        antenna.receive(Signal([0,1e-9,2e-9], [0,1,0], Signal.Type.voltage))
        assert antenna._noise_master is None
        antenna.waveforms
        assert antenna._noise_master is not None

    def test_noises_not_recalculated(self, antenna):
        """Test that noise signals aren't recalculated every time"""
        antenna.signals.append(Signal([0,1e-9],[1,1]))
        waveforms1 = antenna.waveforms
        noise_master_1 = antenna._noise_master
        waveforms2 = antenna.waveforms
        noise_master_2 = antenna._noise_master
        assert noise_master_1 == noise_master_2

    def test_no_trigger_no_waveform(self, antenna):
        """Test that signals which don't trigger don't appear in waveforms,
        but do appear in all_waveforms"""
        antenna.trigger = lambda signal: False
        antenna.signals.append(Signal([0,1e-9],[1,1]))
        assert antenna.is_hit == False
        assert antenna.waveforms == []
        assert antenna.all_waveforms != []

    def test_noise_master_generation(self, antenna):
        """Test that _noise_master is generated the first time make_noise is
        called and never again"""
        assert antenna._noise_master is None
        noise = antenna.make_noise(np.linspace(0, 100e-9))
        assert antenna._noise_master is not None
        old_noise_master = antenna._noise_master
        noise = antenna.make_noise(np.linspace(0, 50e-9))
        assert antenna._noise_master == old_noise_master

    def test_noise_master_failure(self):
        """Test that creation of _noise_master fails if not enough values are
        specified"""
        with pytest.raises(ValueError):
            antenna = Antenna(position=[0,0,-250])
            antenna.make_noise([0,1,2])
        with pytest.raises(ValueError):
            antenna = Antenna(position=[0,0,-250], freq_range=[500e6, 750e6])
            antenna.make_noise([0,1,2])
        with pytest.raises(ValueError):
            antenna = Antenna(position=[0,0,-250], freq_range=[500e6, 750e6],
                              temperature=300)
            antenna.make_noise([0,1,2])
        with pytest.raises(ValueError):
            antenna = Antenna(position=[0,0,-250], freq_range=[500e6, 750e6],
                              resistance=100)
            antenna.make_noise([0,1,2])

    def test_full_waveform(self, triggerable_antenna):
        """Test that full_waveform incorporates all waveforms (even untriggered)"""
        triggerable_antenna.signals.append(Signal([1], [2]))
        triggerable_antenna.signals.append(Signal([0], [0.5]))
        triggerable_antenna.signals.append(Signal([0, 1, 2], [0.1, 0.1, 0.1]))
        full = triggerable_antenna.full_waveform([-1, 0, 1, 2, 3])
        assert np.array_equal(full.values, [0, 0.6, 2.1, 0.1, 0])



@pytest.fixture
def dipole():
    """Fixture for forming basic DipoleAntenna object"""
    return DipoleAntenna(name="ant", position=[0,0,-250],
                         center_frequency=250e6, bandwidth=300e6,
                         temperature=300, resistance=100, orientation=[0,0,1],
                         trigger_threshold=75e-6)


class TestDipoleAntenna:
    """Tests for DipoleAntenna class"""
    def test_creation(self, dipole):
        """Test that the antenna's creation goes as expected"""
        assert dipole.name == "ant"
        assert np.array_equal(dipole.position, [0,0,-250])
        assert np.array_equal(dipole.z_axis, [0,0,1])
        assert dipole.x_axis[2] == 0
        assert dipole.antenna_factor == pytest.approx(2 * 250e6
                                                      / scipy.constants.c)
        assert dipole.efficiency == 1
        assert np.array_equal(dipole.freq_range, [100e6, 400e6])
        assert dipole.temperature == 300
        assert dipole.resistance == 100
        assert dipole.threshold == 75e-6
        assert dipole.noisy

    @pytest.mark.parametrize("freq", np.arange(50, 500, 50)*1e6)
    def test_frequency_response(self, dipole, freq):
        """Test that the frequency response of the dipole antenna is as
        expected"""
        response = dipole.frequency_response(freq)
        db_point = 1/np.sqrt(2)
        if (freq==pytest.approx(dipole.freq_range[0])
                or freq==pytest.approx(dipole.freq_range[1])):
            assert np.abs(response) == pytest.approx(db_point, rel=0.01)
        elif freq>dipole.freq_range[0] and freq<dipole.freq_range[1]:
            assert np.abs(response) > db_point
            assert np.abs(response) <= 1
        else:
            assert np.abs(response) < db_point

    @pytest.mark.parametrize("theta", np.linspace(0, np.pi, 7))
    @pytest.mark.parametrize("phi", np.linspace(0, 2*np.pi, 13))
    def test_directional_gain(self, dipole, theta, phi):
        """Test that the directional gain of the dipole antenna goes as sin"""
        assert dipole.directional_gain(theta, phi) == pytest.approx(np.sin(theta))

    @pytest.mark.parametrize("x", np.linspace(0, 1, 3))
    @pytest.mark.parametrize("y", np.linspace(0, 1, 3))
    @pytest.mark.parametrize("z", np.linspace(0, 1, 11))
    def test_polarization_gain(self, dipole, x, y, z):
        """Test that the polarization gain of the dipole antenna goes as the
        dot product of the antenna axis with the polarization direction
        (i.e. the z-component)"""
        assert dipole.polarization_gain((x,y,z)) == pytest.approx(z)

    def test_trigger(self, dipole):
        dipole.noisy = False
        assert not dipole.is_hit
        dipole.signals.append(Signal([0,1e-9], [0,dipole.threshold*1.01]))
        assert dipole.is_hit

