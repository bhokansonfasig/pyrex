"""File containing tests of pyrex particle module"""

import pytest

from config import SEED

from pyrex.particle import (Event, Particle, Interaction,
                            GQRSInteraction, CTWInteraction)

import numpy as np



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
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                     interaction_model=Interaction)
        assert p.id == Particle.Type.electron
        p = Particle(particle_id=-12,
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.electron_antineutrino
        p = Particle(particle_id="nu_mu",
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)
        assert p.id == Particle.Type.muon_neutrino
        p = Particle(particle_id=None,
                     vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                     interaction_model=Interaction)
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
                          energy=1e9, interaction_model=Interaction)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
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
                          energy=1e9, interaction_model=Interaction)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        with pytest.raises(ValueError):
            event.add_children(child1, [child2])

    def test_get_children(self, particle):
        """Test the ability to retrieve children of a particle"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
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
                          energy=1e9, interaction_model=Interaction)
        with pytest.raises(ValueError):
            event.get_children(child1)

    def test_get_parent(self, particle):
        """Test the ability to retrieve parent of a particle"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        event.add_children(particle, [child1, child2])
        assert event.get_parent(child1) == particle
        assert event.get_parent(child2) == particle
        assert event.get_parent(particle) is None

    def test_get_parent_fails(self, particle):
        """Test that get_parent fails if the parent isn't in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        with pytest.raises(ValueError):
            event.get_parent(child1)

    def test_get_from_level(self, particle):
        """Test the ability to get particles from a level in the tree"""
        event = Event(particle)
        child1 = Particle(particle_id=Particle.Type.electron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
        child2 = Particle(particle_id=Particle.Type.positron,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9, interaction_model=Interaction)
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



# Cross section and inelasticity data from Tables 1 & 2 of GQRS
# (https://arxiv.org/pdf/hep-ph/9512364.pdf)
GQRS_sigma_CC_NU = [(1e1,  0.777e-37), (1e2,  0.697e-36), (1e3,  0.625e-35),
                    (1e4,  0.454e-34), (1e5,  0.196e-33), (1e6,  0.611e-33),
                    (1e7,  0.176e-32), (1e8,  0.478e-32), (1e9,  0.123e-31),
                    (1e10, 0.301e-31), (1e11, 0.706e-31), (1e12, 0.159e-30)]

GQRS_sigma_NC_NU = [(1e1,  0.242e-37), (1e2,  0.217e-36), (1e3,  0.199e-35),
                    (1e4,  0.155e-34), (1e5,  0.745e-34), (1e6,  0.252e-33),
                    (1e7,  0.748e-33), (1e8,  0.207e-32), (1e9,  0.540e-32),
                    (1e10, 0.134e-31), (1e11, 0.316e-31), (1e12, 0.715e-31)]

GQRS_ybar_CC_NU = [#(1e1,  0.483), (1e2,  0.477), (1e3,  0.472),
                   #(1e4,  0.426), (1e5,  0.332), (1e6,  0.273),
                   (1e7,  0.250), (1e8,  0.237), (1e9,  0.225),
                   (1e10, 0.216), (1e11, 0.208), (1e12, 0.207)]

GQRS_ybar_NC_NU = [#(1e1,  0.474), (1e2,  0.470), (1e3,  0.467),
                   #(1e4,  0.428), (1e5,  0.341), (1e6,  0.279),
                   (1e7,  0.254), (1e8,  0.239), (1e9,  0.227),
                   (1e10, 0.217), (1e11, 0.210), (1e12, 0.207)]

GQRS_sigma_CC_NUBAR = [(1e1,  0.368e-37), (1e2,  0.349e-36), (1e3,  0.338e-35),
                       (1e4,  0.292e-34), (1e5,  0.162e-33), (1e6,  0.582e-33),
                       (1e7,  0.174e-32), (1e8,  0.477e-32), (1e9,  0.123e-31),
                       (1e10, 0.301e-31), (1e11, 0.706e-31), (1e12, 0.159e-30)]

GQRS_sigma_NC_NUBAR = [(1e1,  0.130e-37), (1e2,  0.122e-36), (1e3,  0.120e-35),
                       (1e4,  0.106e-34), (1e5,  0.631e-34), (1e6,  0.241e-33),
                       (1e7,  0.742e-33), (1e8,  0.207e-32), (1e9,  0.540e-32),
                       (1e10, 0.134e-31), (1e11, 0.316e-31), (1e12, 0.715e-31)]

GQRS_ybar_CC_NUBAR = [#(1e1,  0.333), (1e2,  0.340), (1e3,  0.354),
                      #(1e4,  0.345), (1e5,  0.301), (1e6,  0.266),
                      (1e7,  0.249), (1e8,  0.237), (1e9,  0.225),
                      (1e10, 0.216), (1e11, 0.208), (1e12, 0.205)]

GQRS_ybar_NC_NUBAR = [#(1e1,  0.350), (1e2,  0.354), (1e3,  0.368),
                      #(1e4,  0.358), (1e5,  0.313), (1e6,  0.273),
                      (1e7,  0.253), (1e8,  0.239), (1e9,  0.227),
                      (1e10, 0.217), (1e11, 0.210), (1e12, 0.207)]


@pytest.fixture
def GQRS_cc_nu():
    """Fixture for forming a neutrino with charged-current GQRSInteraction"""
    return Particle(particle_id=Particle.Type.electron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=GQRSInteraction,
                    interaction_type=Interaction.Type.charged_current)

@pytest.fixture
def GQRS_nc_nu():
    """Fixture for forming a neutrino with neutral-current GQRSInteraction"""
    return Particle(particle_id=Particle.Type.electron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=GQRSInteraction,
                    interaction_type=Interaction.Type.neutral_current)

@pytest.fixture
def GQRS_cc_nubar():
    """Fixture for forming a neutrino with charged-current GQRSInteraction"""
    return Particle(particle_id=Particle.Type.antielectron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=GQRSInteraction,
                    interaction_type=Interaction.Type.charged_current)

@pytest.fixture
def GQRS_nc_nubar():
    """Fixture for forming a neutrino with neutral-current GQRSInteraction"""
    return Particle(particle_id=Particle.Type.antielectron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=GQRSInteraction,
                    interaction_type=Interaction.Type.neutral_current)

class TestGQRSInteraction:
    """Tests for GQRSInteraction class"""
    def test_creation(self, GQRS_cc_nu):
        """Test initialization of interaction"""
        interaction = GQRSInteraction(GQRS_cc_nu)
        assert interaction.particle == GQRS_cc_nu
        assert isinstance(interaction.kind, Interaction.Type)
        assert isinstance(interaction.inelasticity, float)

        interaction = GQRSInteraction(GQRS_cc_nu,
                                      kind=Interaction.Type.neutral_current)
        assert interaction.kind == Interaction.Type.neutral_current

    def test_kind_coercion(self, GQRS_cc_nu):
        """Test that the interaction kind can be set by enum value, int, or str"""
        interaction = GQRSInteraction(GQRS_cc_nu)
        interaction.kind = Interaction.Type.charged_current
        assert interaction.kind == Interaction.Type.charged_current
        interaction = GQRSInteraction(GQRS_cc_nu)
        interaction.kind = 0
        assert interaction.kind == Interaction.Type.undefined
        interaction = GQRSInteraction(GQRS_cc_nu)
        interaction.kind = "nc"
        assert interaction.kind == Interaction.Type.neutral_current
        interaction = GQRSInteraction(GQRS_cc_nu)
        interaction.kind = None
        assert interaction.kind == Interaction.Type.undefined

    def test_choose_interaction(self, GQRS_cc_nu):
        """Test that choose_interaction method properly samples the current ratio"""
        np.random.seed(SEED)
        interaction = GQRSInteraction(GQRS_cc_nu)
        int_types = []
        for _ in range(10000):
            int_types.append(interaction.choose_interaction().value-1)
        assert np.mean(int_types) == pytest.approx(1-0.6865254, abs=0.01)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_CC_NU)
    def test_cross_section_cc_nu(self, energy, sigma):
        """Test that cross_section attribute is correct for
        charged-current neutrino interactions"""
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        assert cc_nu.interaction.cross_section == pytest.approx(sigma, rel=0.0001)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_NC_NU)
    def test_cross_section_nc_nu(self, energy, sigma):
        """Test that cross_section attribute is correct for
        neutral-current neutrino interactions"""
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        assert nc_nu.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_CC_NU)
    def test_total_cross_section_nu(self, energy, sigma):
        """Test that total_cross_section attribute is correct for
        neutrino interactions"""
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        undef_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.undefined
        )
        assert (cc_nu.interaction.total_cross_section ==
                nc_nu.interaction.total_cross_section)
        assert (cc_nu.interaction.cross_section +
                nc_nu.interaction.cross_section ==
                pytest.approx(cc_nu.interaction.total_cross_section))
        assert (undef_nu.interaction.total_cross_section ==
                cc_nu.interaction.total_cross_section)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_CC_NUBAR)
    def test_cross_section_cc_nubar(self, energy, sigma):
        """Test that cross_section attribute is correct for
        charged-current antineutrino interactions"""
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        assert cc_nubar.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_NC_NUBAR)
    def test_cross_section_nc_nubar(self, energy, sigma):
        """Test that cross_section method attribute is correct for
        neutral-current antineutrino interactions"""
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        assert nc_nubar.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", GQRS_sigma_CC_NU)
    def test_total_cross_section_nubar(self, energy, sigma):
        """Test that total_cross_section attribute is correct for
        antineutrino interactions"""
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        undef_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.undefined
        )
        assert (cc_nubar.interaction.total_cross_section ==
                nc_nubar.interaction.total_cross_section)
        assert (cc_nubar.interaction.cross_section +
                nc_nubar.interaction.cross_section ==
                pytest.approx(cc_nubar.interaction.total_cross_section))
        assert (undef_nubar.interaction.total_cross_section ==
                cc_nubar.interaction.total_cross_section)

    @pytest.mark.parametrize("energy,ybar", GQRS_ybar_CC_NU)
    def test_choose_inelasticity_cc_nu(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for charged-current neutrino interactions"""
        np.random.seed(SEED)
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        ys = []
        for _ in range(10000):
            ys.append(cc_nu.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", GQRS_ybar_NC_NU)
    def test_choose_inelasticity_nc_nu(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for neutral-current neutrino interactions"""
        np.random.seed(SEED)
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        ys = []
        for _ in range(10000):
            ys.append(nc_nu.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", GQRS_ybar_CC_NUBAR)
    def test_choose_inelasticity_cc_nubar(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for charged-current antineutrino interactions"""
        np.random.seed(SEED)
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        ys = []
        for _ in range(10000):
            ys.append(cc_nubar.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", GQRS_ybar_NC_NUBAR)
    def test_choose_inelasticity_nc_nubar(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for neutral-current antineutrino interactions"""
        np.random.seed(SEED)
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=GQRSInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        ys = []
        for _ in range(10000):
            ys.append(nc_nubar.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)



# Cross section data from Tables 1 & 2 of CTW
# (https://arxiv.org/pdf/1102.0691.pdf)
CTW_sigma_CC_NU = [(1.0e4,  0.48e-34), (2.5e4,  0.93e-34), (6.0e4,  0.16e-33),
                   (1.0e5,  0.22e-33), (2.5e5,  0.36e-33), (6.0e5,  0.56e-33),
                   (1.0e6,  0.72e-33), (2.5e6,  0.11e-32), (6.0e6,  0.16e-32),
                   (1.0e7,  0.20e-32), (2.5e7,  0.29e-32), (6.0e7,  0.40e-32),
                   (1.0e8,  0.48e-32), (2.5e8,  0.67e-32), (6.0e8,  0.91e-32),
                   (1.0e9,  0.11e-31), (2.5e9,  0.14e-31), (6.0e9,  0.19e-31),
                   (1.0e10, 0.22e-31), (2.5e10, 0.29e-31), (6.0e10, 0.37e-31),
                   (1.0e11, 0.43e-31), (2.5e11, 0.56e-31), (6.0e11, 0.72e-31),
                   (1.0e12, 0.83e-31)]

CTW_sigma_NC_NU = [(1.0e4,  0.16e-34), (2.5e4,  0.32e-34), (6.0e4,  0.57e-34),
                   (1.0e5,  0.78e-34), (2.5e5,  0.13e-33), (6.0e5,  0.21e-33),
                   (1.0e6,  0.27e-33), (2.5e6,  0.41e-33), (6.0e6,  0.61e-33),
                   (1.0e7,  0.76e-33), (2.5e7,  0.11e-32), (6.0e7,  0.16e-32),
                   (1.0e8,  0.19e-32), (2.5e8,  0.27e-32), (6.0e8,  0.36e-32),
                   (1.0e9,  0.43e-32), (2.5e9,  0.58e-32), (6.0e9,  0.77e-32),
                   (1.0e10, 0.90e-32), (2.5e10, 0.12e-31), (6.0e10, 0.15e-31),
                   (1.0e11, 0.18e-31), (2.5e11, 0.23e-31), (6.0e11, 0.30e-31),
                   (1.0e12, 0.35e-31)]

# Inelasticity data from internal calculation
# (i.e. just to confirm no change to code behavior)
CTW_ybar_CC_NU = [(1e4,  0.345), (1e5,  0.313), (1e6,  0.284),
                  (1e7,  0.253), (1e8,  0.229), (1e9,  0.212),
                  (1e10, 0.200), (1e11, 0.191), (1e12, 0.185)]

CTW_ybar_NC_NU = [(1e4,  0.337), (1e5,  0.305), (1e6,  0.275),
                  (1e7,  0.244), (1e8,  0.220), (1e9,  0.202),
                  (1e10, 0.189), (1e11, 0.180), (1e12, 0.174)]

CTW_sigma_CC_NUBAR = [(1.0e4,  0.29e-34), (2.5e4,  0.63e-34), (6.0e4,  0.12e-33),
                      (1.0e5,  0.17e-33), (2.5e5,  0.30e-33), (6.0e5,  0.49e-33),
                      (1.0e6,  0.63e-33), (2.5e6,  0.98e-33), (6.0e6,  0.15e-32),
                      (1.0e7,  0.18e-32), (2.5e7,  0.26e-32), (6.0e7,  0.37e-32),
                      (1.0e8,  0.45e-32), (2.5e8,  0.62e-32), (6.0e8,  0.84e-32),
                      (1.0e9,  0.99e-32), (2.5e9,  0.13e-31), (6.0e9,  0.17e-31),
                      (1.0e10, 0.20e-31), (2.5e10, 0.27e-31), (6.0e10, 0.35e-31),
                      (1.0e11, 0.40e-31), (2.5e11, 0.52e-31), (6.0e11, 0.66e-31),
                      (1.0e12, 0.77e-31)]

CTW_sigma_NC_NUBAR = [(1.0e4,  0.11e-34), (2.5e4,  0.24e-34), (6.0e4,  0.47e-34),
                      (1.0e5,  0.67e-34), (2.5e5,  0.12e-33), (6.0e5,  0.20e-33),
                      (1.0e6,  0.26e-33), (2.5e6,  0.40e-33), (6.0e6,  0.60e-33),
                      (1.0e7,  0.76e-33), (2.5e7,  0.11e-32), (6.0e7,  0.16e-32),
                      (1.0e8,  0.19e-32), (2.5e8,  0.27e-32), (6.0e8,  0.36e-32),
                      (1.0e9,  0.43e-32), (2.5e9,  0.58e-32), (6.0e9,  0.77e-32),
                      (1.0e10, 0.90e-32), (2.5e10, 0.12e-31), (6.0e10, 0.15e-31),
                      (1.0e11, 0.18e-31), (2.5e11, 0.23e-31), (6.0e11, 0.30e-31),
                      (1.0e12, 0.35e-31)]

CTW_ybar_CC_NUBAR = [(1e4,  0.316), (1e5,  0.285), (1e6,  0.257),
                     (1e7,  0.228), (1e8,  0.205), (1e9,  0.187),
                     (1e10, 0.175), (1e11, 0.166), (1e12, 0.161)]

CTW_ybar_NC_NUBAR = [(1e4,  0.337), (1e5,  0.305), (1e6,  0.275),
                     (1e7,  0.244), (1e8,  0.220), (1e9,  0.202),
                     (1e10, 0.189), (1e11, 0.180), (1e12, 0.174)]


@pytest.fixture
def CTW_cc_nu():
    """Fixture for forming a neutrino with charged-current CTWInteraction"""
    return Particle(particle_id=Particle.Type.electron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=CTWInteraction,
                    interaction_type=Interaction.Type.charged_current)

@pytest.fixture
def CTW_nc_nu():
    """Fixture for forming a neutrino with neutral-current CTWInteraction"""
    return Particle(particle_id=Particle.Type.electron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=CTWInteraction,
                    interaction_type=Interaction.Type.neutral_current)

@pytest.fixture
def CTW_cc_nubar():
    """Fixture for forming a neutrino with charged-current CTWInteraction"""
    return Particle(particle_id=Particle.Type.antielectron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=CTWInteraction,
                    interaction_type=Interaction.Type.charged_current)

@pytest.fixture
def CTW_nc_nubar():
    """Fixture for forming a neutrino with neutral-current CTWInteraction"""
    return Particle(particle_id=Particle.Type.antielectron_neutrino,
                    vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9,
                    interaction_model=CTWInteraction,
                    interaction_type=Interaction.Type.neutral_current)

class TestCTWInteraction:
    """Tests for CTWInteraction class"""
    def test_creation(self, CTW_cc_nu):
        """Test initialization of interaction"""
        interaction = CTWInteraction(CTW_cc_nu)
        assert interaction.particle == CTW_cc_nu
        assert isinstance(interaction.kind, Interaction.Type)
        assert isinstance(interaction.inelasticity, float)

        interaction = CTWInteraction(CTW_cc_nu,
                                      kind=Interaction.Type.neutral_current)
        assert interaction.kind == Interaction.Type.neutral_current

    def test_kind_coercion(self, CTW_cc_nu):
        """Test that the interaction kind can be set by enum value, int, or str"""
        interaction = CTWInteraction(CTW_cc_nu)
        interaction.kind = Interaction.Type.charged_current
        assert interaction.kind == Interaction.Type.charged_current
        interaction = CTWInteraction(CTW_cc_nu)
        interaction.kind = 0
        assert interaction.kind == Interaction.Type.undefined
        interaction = CTWInteraction(CTW_cc_nu)
        interaction.kind = "nc"
        assert interaction.kind == Interaction.Type.neutral_current
        interaction = CTWInteraction(CTW_cc_nu)
        interaction.kind = None
        assert interaction.kind == Interaction.Type.undefined

    @pytest.mark.parametrize("energy", np.logspace(4, 12, num=25))
    def test_choose_interaction(self, energy):
        """Test that choose_interaction method properly samples the current ratio"""
        np.random.seed(SEED)
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction
        )
        int_types = []
        for _ in range(10000):
            int_types.append(cc_nu.interaction.choose_interaction().value-1)
        fraction = 0.252162 + 0.0256*np.log(np.log10(energy)-1.76)
        assert np.mean(int_types) == pytest.approx(fraction, rel=0.05)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_CC_NU)
    def test_cross_section_cc_nu(self, energy, sigma):
        """Test that cross_section attribute is correct for
        charged-current neutrino interactions"""
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        assert cc_nu.interaction.cross_section == pytest.approx(sigma, rel=0.0001)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_NC_NU)
    def test_cross_section_nc_nu(self, energy, sigma):
        """Test that cross_section attribute is correct for
        neutral-current neutrino interactions"""
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        assert nc_nu.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_CC_NU)
    def test_total_cross_section_nu(self, energy, sigma):
        """Test that total_cross_section attribute is correct for
        neutrino interactions"""
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        undef_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.undefined
        )
        assert (cc_nu.interaction.total_cross_section ==
                nc_nu.interaction.total_cross_section)
        assert (cc_nu.interaction.cross_section +
                nc_nu.interaction.cross_section ==
                pytest.approx(cc_nu.interaction.total_cross_section))
        assert (undef_nu.interaction.total_cross_section ==
                cc_nu.interaction.total_cross_section)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_CC_NUBAR)
    def test_cross_section_cc_nubar(self, energy, sigma):
        """Test that cross_section attribute is correct for
        charged-current antineutrino interactions"""
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        assert cc_nubar.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_NC_NUBAR)
    def test_cross_section_nc_nubar(self, energy, sigma):
        """Test that cross_section method attribute is correct for
        neutral-current antineutrino interactions"""
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        assert nc_nubar.interaction.cross_section == pytest.approx(sigma, rel=0.001)

    @pytest.mark.parametrize("energy,sigma", CTW_sigma_CC_NU)
    def test_total_cross_section_nubar(self, energy, sigma):
        """Test that total_cross_section attribute is correct for
        antineutrino interactions"""
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        undef_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.undefined
        )
        assert (cc_nubar.interaction.total_cross_section ==
                nc_nubar.interaction.total_cross_section)
        assert (cc_nubar.interaction.cross_section +
                nc_nubar.interaction.cross_section ==
                pytest.approx(cc_nubar.interaction.total_cross_section))
        assert (undef_nubar.interaction.total_cross_section ==
                cc_nubar.interaction.total_cross_section)

    @pytest.mark.parametrize("energy,ybar", CTW_ybar_CC_NU)
    def test_choose_inelasticity_cc_nu(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for charged-current neutrino interactions"""
        np.random.seed(SEED)
        cc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        ys = []
        for _ in range(10000):
            ys.append(cc_nu.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", CTW_ybar_NC_NU)
    def test_choose_inelasticity_nc_nu(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for neutral-current neutrino interactions"""
        np.random.seed(SEED)
        nc_nu = Particle(
            particle_id=Particle.Type.electron_neutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        ys = []
        for _ in range(10000):
            ys.append(nc_nu.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", CTW_ybar_CC_NUBAR)
    def test_choose_inelasticity_cc_nubar(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for charged-current antineutrino interactions"""
        np.random.seed(SEED)
        cc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.charged_current
        )
        ys = []
        for _ in range(10000):
            ys.append(cc_nubar.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)

    @pytest.mark.parametrize("energy,ybar", CTW_ybar_NC_NUBAR)
    def test_choose_inelasticity_nc_nubar(self, energy, ybar):
        """Test that choose_inelasticity method properly samples the inelasticity
        distribution for neutral-current antineutrino interactions"""
        np.random.seed(SEED)
        nc_nubar = Particle(
            particle_id=Particle.Type.electron_antineutrino,
            vertex=[100, 200, -500], direction=[0, 0, 1], energy=energy,
            interaction_model=CTWInteraction,
            interaction_type=Interaction.Type.neutral_current
        )
        ys = []
        for _ in range(10000):
            ys.append(nc_nubar.interaction.choose_inelasticity())
        assert np.mean(ys) == pytest.approx(ybar, rel=0.2)
