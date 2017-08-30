"""File containing tests of pyrex particle module"""

import pytest

from pyrex.particle import CC_NU #, Particle, ShadowGenerator

# import numpy as np


CC_NU_cross_sections = [(1,   2.690e-36), (10,  6.788e-36), (100, 1.713e-35),
                        (1e3, 4.323e-35), (1e4, 1.091e-34), (1e5, 2.753e-34),
                        (1e6, 6.946e-34), (1e7, 1.753e-33), (1e8, 4.423e-33)]

CC_NU_interaction_lengths = [(1,   6.175e11), (10,  2.447e11),
                             (100, 9.697e10), (1e3, 3.842e10),
                             (1e4, 1.523e10), (1e5, 6.035e9),
                             (1e6, 2.391e9),  (1e7, 9.477e8),
                             (1e8, 3.755e8)]


class TestCC_NU:
    """Tests for CC_NU object"""
    @pytest.mark.parametrize("energy,cross_section",
                             CC_NU_cross_sections)
    def test_cross_section(self, energy, cross_section):
        """Test that the cross section for CC_NU is as expected within 2%"""
        assert (CC_NU.cross_section(energy) ==
                pytest.approx(cross_section, rel=0.02, abs=1e-36))

    @pytest.mark.parametrize("energy,interaction_length",
                             CC_NU_interaction_lengths)
    def test_interaction_length(self, energy, interaction_length):
        """Test that the interaction length for CC_NU is as expected within 2%"""
        assert (CC_NU.interaction_length(energy) ==
                pytest.approx(interaction_length, rel=0.02))
