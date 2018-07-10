"""File containing tests of pyrex particle module"""

import pytest

from config import SEED

from pyrex.particle import (Event, Particle, Interaction,
                            GQRSInteraction, CTWInteraction)

import numpy as np


CC_NU_cross_sections = [(1,   2.690e-36), (10,  6.788e-36), (100, 1.713e-35),
                        (1e3, 4.323e-35), (1e4, 1.091e-34), (1e5, 2.753e-34),
                        (1e6, 6.946e-34), (1e7, 1.753e-33), (1e8, 4.423e-33)]

CC_NU_interaction_lengths = [(1,   6.175e11), (10,  2.447e11),
                             (100, 9.697e10), (1e3, 3.842e10),
                             (1e4, 1.523e10), (1e5, 6.035e9),
                             (1e6, 2.391e9),  (1e7, 9.477e8),
                             (1e8, 3.755e8)]


# class TestCC_NU:
#     """Tests for CC_NU object"""
#     @pytest.mark.parametrize("energy,cross_section",
#                              CC_NU_cross_sections)
#     def test_cross_section(self, energy, cross_section):
#         """Test that the cross section for CC_NU is as expected within 2%"""
#         assert (CC_NU.cross_section(energy) ==
#                 pytest.approx(cross_section, rel=0.02, abs=1e-36))

#     @pytest.mark.parametrize("energy,interaction_length",
#                              CC_NU_interaction_lengths)
#     def test_interaction_length(self, energy, interaction_length):
#         """Test that the interaction length for CC_NU is as expected within 2%"""
#         assert (CC_NU.interaction_length(energy) ==
#                 pytest.approx(interaction_length, rel=0.02))


# TODO: Add tests for all neutrino interactions based on data in papers


@pytest.fixture
def particle():
    """Fixture for forming basic Particle object"""
    return Particle(particle_id=Particle.Type.electron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)

class TestParticle:
    """Tests for Particle class"""
    def test_creation(self, particle):
        """Test initialization of particle"""
        assert particle.id == Particle.Type.electron_neutrino
        assert np.array_equal(particle.vertex, [100, 200, -500])
        assert np.array_equal(particle.direction, [0, 0, 1])
        assert particle.energy == 1e9
        assert isinstance(particle.interaction, Interaction)
        assert particle.weight == 1

    def test_direction_normalization(self):
        """Test that the particle direction is automatically normalized"""
        direction = [0, 1, 1]
        p = Particle(particle_id=Particle.Type.electron_neutrino,
                     vertex=[100, 200, -500], direction=direction, energy=1e9)
        assert np.array_equal(p.direction, direction/np.linalg.norm(direction))

    def test_id_coercion(self):
        """Test that the particle id can be set by enum value, int, or str"""
        p = Particle(particle_id=Particle.Type.electron,
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.electron
        p = Particle(particle_id=-12,
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.electron_antineutrino
        p = Particle(particle_id="nu_mu",
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.muon_neutrino
        p = Particle(particle_id=None,
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.undefined

    def test_interaction_type_setting(self):
        """Test that the interaction type can be set during construction"""
        p = Particle(particle_id="nu_e", vertex=[100, 200, -500],
                     direction=[0, 0, 1], energy=1e9,
                     interaction_type=Interaction.Type.charged_current)
        assert p.interaction.kind == Interaction.Type.charged_current



class TestEvent:
    """Tests for Event class"""
    def test_creation(self, particle):
        """Test initialization of event"""
        event = Event(particle)
        assert event.roots[0] == particle
        assert len(event.roots) == 1
        assert len(event) == 1
        for p in event:
            assert p == particle

    def test_add_children(self, particle):
        """Test the ability to add children to a particle"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(particle, [child1, child2])
        assert len(event.roots) == 1
        assert len(event) == 3
        all_particles = [particle, child1, child2]
        for p in event:
            assert p in all_particles
            all_particles.remove(p)
        assert all_particles == []
        child3 = Particle(particle_id=Particle.Type.electron_antineutrino,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(child1, child3)
        assert len(event) == 4
        all_particles = [particle, child1, child2, child3]
        for p in event:
            assert p in all_particles
            all_particles.remove(p)
        assert all_particles == []

    def test_add_children_fails(self, particle):
        """Test that add_children fails if the parent isn't in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        with pytest.raises(ValueError):
            event.add_children(child1, [child2])

    def test_get_children(self, particle):
        """Test the ability to retrieve children of a particle"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(particle, [child1, child2])
        expected_children = [child1, child2]
        for child in event.get_children(particle):
            assert child in expected_children
            expected_children.remove(child)
        assert expected_children == []
        assert event.get_children(child1) == []

    def test_get_children_fails(self, particle):
        """Test that get_children fails if the parent isn't in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        with pytest.raises(ValueError):
            event.get_children(child1)

    def test_get_parent(self, particle):
        """Test the ability to retrieve parent of a particle"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(particle, [child1, child2])
        assert event.get_parent(child1) == particle
        assert event.get_parent(child2) == particle
        assert event.get_parent(particle) is None

    def test_get_parent_fails(self, particle):
        """Test that get_parent fails if the parent isn't in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        with pytest.raises(ValueError):
            event.get_parent(child1)

    def test_get_from_level(self, particle):
        """Test the ability to get particles from a level in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(particle, [child1, child2])
        expected_level_0 = [particle]
        expected_level_1 = [child1, child2]
        for p in event.get_from_level(0):
            assert p in expected_level_0
            expected_level_0.remove(p)
        assert expected_level_0 == []
        for p in event.get_from_level(1):
            assert p in expected_level_1
            expected_level_1.remove(p)
        assert expected_level_1 == []
        assert event.get_from_level(2) == []
        child3 = Particle(particle_id=Particle.Type.electron_antineutrino,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9)
        event.add_children(child1, child3)
        expected_level_0 = [particle]
        expected_level_1 = [child1, child2]
        expected_level_2 = [child3]
        for p in event.get_from_level(0):
            assert p in expected_level_0
            expected_level_0.remove(p)
        assert expected_level_0 == []
        for p in event.get_from_level(1):
            assert p in expected_level_1
            expected_level_1.remove(p)
        assert expected_level_1 == []
        for p in event.get_from_level(2):
            assert p in expected_level_2
            expected_level_2.remove(p)
        assert expected_level_2 == []
