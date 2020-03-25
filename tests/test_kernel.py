"""File containing tests of pyrex kernel module"""

import pytest

from pyrex.askaryan import AskaryanSignal
from pyrex.antenna import Antenna
from pyrex.ice_model import ice
from pyrex.ray_tracing import RayTracer
from pyrex.particle import Particle, Event
from pyrex.generation import ListGenerator
from pyrex.kernel import EventKernel

import numpy as np



@pytest.fixture
def kernel():
    """Fixture for forming basic EventKernel object"""
    gen = ListGenerator(Event(Particle(particle_id="electron_neutrino",
                                       vertex=[100, 200, -500],
                                       direction=[0, 0, 1],
                                       energy=1e9)))
    return EventKernel(generator=gen, antennas=[Antenna(position=(0, 0, -100),
                                                        noisy=False)])


class TestEventKernel:
    """Tests for EventKernel class"""
    def test_creation(self, kernel):
        """Test initialization of kernel"""
        assert isinstance(kernel.gen, ListGenerator)
        assert len(kernel.antennas) == 1
        assert isinstance(kernel.antennas[0], Antenna)
        assert kernel.ice == ice
        assert kernel.ray_tracer == RayTracer
        assert kernel.signal_model == AskaryanSignal
        assert np.array_equal(kernel.signal_times,
                              np.linspace(-50e-9, 50e-9, 2000, endpoint=False))

    def test_event(self, kernel):
        """Test that the event method runs smoothly"""
        event = kernel.event()
        particle = event.roots[0]
        assert np.array_equal(particle.vertex, [100, 200, -500])
        assert np.array_equal(particle.direction, [0, 0, 1])
        assert particle.energy == 1e9
        for ant in kernel.antennas:
            assert len(ant.signals) == 2
