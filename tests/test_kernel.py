"""File containing tests of pyrex kernel module"""

import pytest

from pyrex.antenna import Antenna
from pyrex.ice_model import IceModel
from pyrex.particle import Particle, ListGenerator
from pyrex.kernel import EventKernel

import numpy as np



@pytest.fixture
def kernel():
    """Fixture for forming basic EventKernel object"""
    gen = ListGenerator(Particle(vertex=[100, 200, -500],
                                 direction=[0, 0, 1],
                                 energy=1e9))
    return EventKernel(generator=gen, antennas=[Antenna(position=(0, 0, -100),
                                                        noisy=False)])


class TestEventKernel:
    """Tests for EventKernel class"""
    def test_creation(self, kernel):
        """Test initialization of kernel"""
        assert isinstance(kernel.gen, ListGenerator)
        assert len(kernel.antennas) == 1
        assert isinstance(kernel.antennas[0], Antenna)
        assert kernel.ice == IceModel

    def test_event(self, kernel):
        """Test that the event method runs smoothly"""
        particle = kernel.event()
        assert np.array_equal(particle.vertex, [100, 200, -500])
        assert np.array_equal(particle.direction, [0, 0, 1])
        assert particle.energy == 1e9
        for ant in kernel.antennas:
            assert len(ant.signals) == 2
