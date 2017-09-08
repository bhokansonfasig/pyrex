"""File containing tests of pyrex ray_tracing module"""

import pytest

from pyrex.ray_tracing import PathFinder
from pyrex.ice_model import AntarcticIce

import numpy as np


@pytest.fixture
def path_finder():
    """Fixture for forming basic PathFinder object"""
    return PathFinder(AntarcticIce, [0,0,-100.], [0,0,-200.])

@pytest.fixture
def bad_path():
    """Fixture for forming PathFinder object whose path doesn't exist"""
    return PathFinder(AntarcticIce, [100,0,-200], [0,0,-200])


path_attenuations = [(1e3, 0.9993676), (1e4, 0.9985931), (1e5, 0.9968715),
                     (1e6, 0.9930505), (1e7, 0.9845992), (1e8, 0.9660472),
                     (1e9, 0.9260033), (1e10, 2.625058e-4)]
# TODO: Confirm sharp drop-off above 1 GHz

class TestPathFinder:
    """Tests for PathFinder class"""
    def test_creation(self, path_finder):
        """Test that the PathFinder's creation goes as expected"""
        assert np.array_equal(path_finder.from_point, [0,0,-100])
        assert np.array_equal(path_finder.to_point,   [0,0,-200])
        assert (isinstance(path_finder.ice, AntarcticIce) or
                issubclass(path_finder.ice, AntarcticIce))

    def test_exists(self, path_finder, bad_path):
        """Test that the exists parameter works as expected"""
        assert path_finder.exists
        assert not bad_path.exists

    def test_exists_not_writable(self, path_finder):
        """Test that the exists parameter is not assignable"""
        with pytest.raises(AttributeError):
            path_finder.exists = False

    def test_emitted_ray(self, path_finder):
        """Test that the emitted_ray property works as expected"""
        assert np.array_equal(path_finder.emitted_ray, [0,0,-1])

    def test_emitted_ray_not_writable(self, path_finder):
        """Test that the emitted_ray parameter is not assignable"""
        with pytest.raises(AttributeError):
            path_finder.emitted_ray = np.array([0,0,1])

    def test_path_length(self, path_finder):
        """Test that the path_length property works as expected"""
        assert path_finder.path_length == pytest.approx(100)

    def test_path_length_not_writable(self, path_finder):
        """Test that the emitted_ray parameter is not assignable"""
        with pytest.raises(AttributeError):
            path_finder.path_length = 0

    def test_time_of_flight(self, path_finder):
        """Test that the detailed time of flight gives the expected value
        within 0.01%"""
        assert (path_finder.time_of_flight(n_steps=10000)
                == pytest.approx(5.643462e-7, rel=0.0001))

    def test_tof(self, path_finder):
        """Test that the tof parameter gives the correct time of flight
        within 1%"""
        assert path_finder.tof == pytest.approx(5.643462e-7, rel=0.01)

    def test_tof_not_writable(self, path_finder):
        """Test that the tof parameter is not assignable"""
        with pytest.raises(AttributeError):
            path_finder.tof = 0

    @pytest.mark.parametrize("frequency,attenuation", path_attenuations)
    def test_attenuation(self, path_finder, frequency, attenuation):
        """Test that detailed attenuation returns the expected values
        within 0.01%"""
        assert (path_finder.attenuation(frequency, n_steps=10000)
                == pytest.approx(attenuation, rel=0.0001))

    # 10 GHz test excluded since it's low value means the test fails
    # FIXME when sharp cutoff above 1 GHz is confirmed
    @pytest.mark.parametrize("frequency,attenuation", path_attenuations[:7])
    def test_attenuation(self, path_finder, frequency, attenuation):
        """Test that attenuation returns the expected values within 1%"""
        assert (path_finder.attenuation(frequency)
                == pytest.approx(attenuation, rel=0.01))
