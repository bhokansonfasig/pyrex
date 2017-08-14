"""File containing tests of pyrex kernel module"""

import pytest

from pyrex.kernel import PathFinder
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


path_attenuations = [(1e-3, 0.99937), (1e-2, 0.99859), (1e-1, 0.99687),
                     (1,    0.99305), (10,   0.98460), (100,  0.96605),
                     (1e3,  0.92601), (1e4,  2.627e-4)]
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
    
    @pytest.mark.parametrize("frequency,attenuation", path_attenuations)
    def test_propagate_ray(self, path_finder, frequency, attenuation):
        """Test that propagate_ray returns the expected values within 1%"""
        atten, tof = path_finder.propagate_ray(frequency)
        assert atten == pytest.approx(attenuation, rel=0.01)
        

