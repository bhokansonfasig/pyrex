"""File containing tests of pyrex particle module"""

import pytest

from config import SEED

from pyrex.particle import (CC_NU, Particle, random_direction,
                            ShadowGenerator, ListGenerator, FileGenerator)

import numpy as np


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


# TODO: Add tests for all neutrino interactions based on data in paper


@pytest.fixture
def particle():
    """Fixture for forming basic Particle object"""
    return Particle(vertex=[100, 200, -500], direction=[0, 0, 1], energy=1e9)

class TestParticle:
    """Tests for Particle class"""
    def test_creation(self, particle):
        """Test initialization of particle"""
        assert np.array_equal(particle.vertex, [100, 200, -500])
        assert np.array_equal(particle.direction, [0, 0, 1])
        assert particle.energy == 1e9

    def test_direction_normalization(self):
        """Test that the particle direction is automatically normalized"""
        direction = [0, 1, 1]
        particle = Particle(vertex=[100, 200, -500], direction=direction, energy=1e9)
        assert np.array_equal(particle.direction, direction/np.linalg.norm(direction))



class Test_random_direction:
    """Tests for random_direction function"""
    def test_unit_vector(self):
        """Test that the direction is a unit vector"""
        np.random.seed(SEED)
        for _ in range(10):
            v = random_direction()
            assert np.linalg.norm(v) == pytest.approx(1)

    def test_uniform(self):
        """Test that the random directions are uniform on the unit sphere"""
        np.random.seed(SEED)
        xs = []
        ys = []
        zs = []
        for _ in range(10000):
            v = random_direction()
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])
        assert np.mean(xs) == pytest.approx(0, abs=0.01)
        assert np.std(xs) == pytest.approx(2/np.sqrt(12), rel=0.01)
        assert np.mean(ys) == pytest.approx(0, abs=0.01)
        assert np.std(ys) == pytest.approx(2/np.sqrt(12), rel=0.01)
        assert np.mean(zs) == pytest.approx(0, abs=0.01)
        assert np.std(zs) == pytest.approx(2/np.sqrt(12), rel=0.01)



class TestShadowGenerator:
    """Tests for ShadowGenerator class"""
    def test_creation(self):
        """Test initialization of ShadowGenerator"""
        generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                    energy=1e9)
        assert generator.dx == 5000
        assert generator.dy == 5000
        assert generator.dz == 3000
        assert generator.energy_generator() == 1e9
        assert generator.count == 0

        generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                    energy=lambda: 1e9)
        assert generator.energy_generator() == 1e9

    def test_creation_failure(self):
        """Test that initialization fails if the energy_generator is not
        a function or a float-like value"""
        with pytest.raises(ValueError):
            generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                        energy=[1e9])

    def test_create_particle(self):
        """Test that create_particle method returns a particle object"""
        np.random.seed(SEED)
        generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                    energy=lambda: 1e9)
        particle = generator.create_particle()
        assert isinstance(particle, Particle)



class TestListGenerator:
    """Tests for ListGenerator class"""
    def test_creation(self, particle):
        """Test initialization of ListGenerator"""
        generator = ListGenerator(particle)
        assert generator.particles[0] == particle
        assert generator.loop
        generator = ListGenerator([particle, particle])
        assert generator.particles[0] == particle
        assert generator.particles[1] == particle
        assert generator.loop

    def test_create_particle(self, particle):
        particle2 = Particle(vertex=[0, 0, 0], direction=[0, 0, -1], energy=1e9)
        generator = ListGenerator([particle, particle2])
        assert generator.create_particle() == particle
        assert generator.create_particle() == particle2

    def test_loop(self, particle):
        """Test that the loop property allows for turning on and off the
        re-iteration of the list of particles"""
        particle2 = Particle(vertex=[0, 0, 0], direction=[0, 0, -1], energy=1e9)
        generator = ListGenerator([particle, particle2], loop=True)
        assert generator.loop
        assert generator.create_particle() == particle
        assert generator.create_particle() == particle2
        assert generator.create_particle() == particle
        assert generator.create_particle() == particle2
        generator = ListGenerator(particle, loop=False)
        assert not generator.loop
        assert generator.create_particle() == particle
        with pytest.raises(StopIteration):
            generator.create_particle()



test_vertices = [(0, 0, 0), (0, 0, -100), (-100, -100, -300), (100, 200, -500)]
test_directions = [(0, 0, -1), (0, 0, 1), (1, 0, 1), (0, 0, 1)]
test_energies = [1e9]*4

@pytest.fixture
def file_gen(tmpdir):
    """Fixture for forming basic FileGenerator object,
    including creating temporary .npz files (once per test)."""
    if not "test_particles_1.npz" in [f.basename for f in tmpdir.listdir()]:
        np.savez(str(tmpdir.join("test_particles_1.npz")),
                 vertices=test_vertices[:2], directions=test_directions[:2],
                 energies=test_energies[:2])
        np.savez(str(tmpdir.join("test_particles_2.npz")),
                 test_vertices[2:], test_directions[2:], test_energies[2:])
    return FileGenerator([str(tmpdir.join("test_particles_1.npz")),
                          str(tmpdir.join("test_particles_2.npz"))])

class TestFileGenerator:
    """Tests for FileGenerator class"""
    def test_creation(self, file_gen, tmpdir):
        """Test initialization of FileGenerator"""
        assert file_gen.files == [str(tmpdir.join("test_particles_1.npz")),
                                  str(tmpdir.join("test_particles_2.npz"))]
        file_gen_2 = FileGenerator(str(tmpdir.join("test_particles_1.npz")))
        assert file_gen_2.files == [str(tmpdir.join("test_particles_1.npz"))]

    def test_create_particle(self, file_gen, tmpdir):
        """Test that create_particle method loops over files correctly.
        Also tests ability to read files without explicit labels since
        test_particles_2.npz is created without explicit labels"""
        for i in range(4):
            particle = file_gen.create_particle()
            expected = Particle(vertex=test_vertices[i],
                                direction=test_directions[i],
                                energy=test_energies[i])
            assert np.array_equal(particle.vertex, expected.vertex)
            assert np.array_equal(particle.direction, expected.direction)
            assert particle.energy == expected.energy
        with pytest.raises(StopIteration):
            file_gen.create_particle()

    def test_bad_files(self, tmpdir):
        """Test that appropriate errors are raised when bad files are passed"""
        np.savez(str(tmpdir.join("bad_particles_1.npz")),
                 some=[(0, 0, 0), (0, 0, -100)], bad=[(0, 0, -1), (0, 0, 1)],
                 keys=[1e9]*2)
        with pytest.raises(KeyError):
            FileGenerator(str(tmpdir.join("bad_particles_1.npz")))
        np.savez(str(tmpdir.join("bad_particles_2.npz")),
                 [(0, 0, 0), (0, 0, -100)], [(0, 0, -1), (0, 0, 1)], [1e9])
        with pytest.raises(ValueError):
            gen = FileGenerator(str(tmpdir.join("bad_particles_2.npz")))
