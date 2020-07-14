"""File containing tests of pyrex askaryan module"""

import pytest

from config import SEED

from pyrex.signals import Signal
from pyrex.askaryan import AskaryanSignal
from pyrex.ice_model import ice
from pyrex.particle import Particle

import numpy as np



@pytest.fixture
def arz_pulse():
    """Example Askaryan pulse from https://arxiv.org/pdf/1106.6283v3.pdf"""
    # Create particle to ensure shower energy is 3e9 GeV
    particle = Particle(particle_id=Particle.Type.electron_neutrino,
                        vertex=(0, 0, -1000), direction=(0, 0, 1), energy=3e9,
                        interaction_type="cc")
    particle.interaction.em_frac = 1
    particle.interaction.had_frac = 0
    n = ice.index(particle.vertex[2])
    cherenkov_angle = np.arcsin(np.sqrt(1 - 1/n**2))
    return AskaryanSignal(times=np.linspace(0, 3e-9, 301),
                          particle=particle,
                          viewing_angle=cherenkov_angle-np.radians(0.3),
                          t0=1e-9)


class TestAskaryanSignal:
    """Tests for AksaryanSignal class"""
    def test_arz_pulse(self, arz_pulse):
        assert arz_pulse.em_energy == 3e9
        assert arz_pulse.had_energy == 0
        assert np.array_equal(arz_pulse.times, np.linspace(0, 3e-9, 301))
        assert arz_pulse.value_type == Signal.Type.field
        # FIXME: Fix the amplitude of Askaryan pulses and use these amplitude tests
        # assert np.max(arz_pulse.values) == pytest.approx(200, rel=0.1)
        # assert np.min(arz_pulse.values) == pytest.approx(-200, rel=0.1)
        peak_to_peak_time = (arz_pulse.times[np.argmin(arz_pulse.values)] -
                             arz_pulse.times[np.argmax(arz_pulse.values)])
        assert peak_to_peak_time == pytest.approx(0.2e-9, abs=0.06e-9)

    # TODO: Add tests for vector_potential, RAC, charge_profile, and max_length methods
