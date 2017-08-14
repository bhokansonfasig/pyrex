"""File containing tests of pyrex particle module"""

import pytest

from pyrex.particle import CC_NU, ShadowGenerator

import numpy as np


CC_NU_cross_sections = [(1,   5.021e-15), (10,  1.267e-14), (100, 3.197e-14),
                        (1e3, 8.068e-14), (1e4, 2.036e-13), (1e5, 5.138e-13),
                        (1e6, 1.296e-12), (1e7, 3.272e-12), (1e8, 8.256e-12)]

CC_NU_interaction_lengths = [(1,   3.309e-10), (10,  1.311e-10),
                             (100, 5.196e-11), (1e3, 2.059e-11),
                             (1e4, 8.159e-12), (1e5, 3.233e-12),
                             (1e6, 1.281e-12), (1e7, 5.077e-13),
                             (1e8, 2.012e-13)]


class TestCC_NU:
    """Tests for CC_NU object"""
    @pytest.mark.parametrize("energy,cross_section",
                             CC_NU_cross_sections)
    def test_cross_section(self, energy, cross_section):
        """Test that the cross section for CC_NU is as expected within 2%"""
        assert (CC_NU.cross_section(energy) ==
                pytest.approx(cross_section, rel=0.02))

    @pytest.mark.parametrize("energy,interaction_length",
                             CC_NU_interaction_lengths)
    def test_interaction_length(self, energy, interaction_length):
        """Test that the interaction length for CC_NU is as expected within 2%"""
        assert (CC_NU.interaction_length(energy) ==
                pytest.approx(interaction_length, rel=0.02))
