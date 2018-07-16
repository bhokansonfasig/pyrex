"""File containing tests of pyrex generation module"""

import pytest

from config import SEED

from pyrex.generation import ShadowGenerator, ListGenerator, FileGenerator
from pyrex.particle import Event, Particle, NeutrinoInteraction

import numpy as np



@pytest.fixture
def generator():
    """Fixture for forming basic ShadowGenerator object"""
    np.random.seed(SEED)
    return ShadowGenerator(dx=5000, dy=5000, dz=3000,
                           energy=1e9)

class TestShadowGenerator:
    """Tests for ShadowGenerator class"""
    def test_creation(self, generator):
        """Test initialization of ShadowGenerator"""
        assert generator.dx == 5000
        assert generator.dy == 5000
        assert generator.dz == 3000
        assert generator.get_energy() == 1e9
        assert np.array_equal(generator.ratio, np.ones(3)/3)
        assert issubclass(generator.interaction_model, NeutrinoInteraction)
        assert generator.count == 0

        generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                    energy=lambda: 1e9)
        assert generator.get_energy() == 1e9

    def test_creation_failure(self):
        """Test that initialization fails if the energy_generator is not
        a function or a float-like value"""
        with pytest.raises(ValueError):
            generator = ShadowGenerator(dx=5000, dy=5000, dz=3000,
                                        energy=[1e9])

    def test_create_event(self, generator):
        """Test that create_event method returns an Event object"""
        event = generator.create_event()
        assert isinstance(event, Event)

    def test_get_vertex(self, generator):
        """Test that get_vertex method uniformly samples in the desired volume"""
        xs = []
        ys = []
        zs = []
        v = generator.get_vertex()
        assert isinstance(v, np.ndarray)
        assert len(v) == 3
        for _ in range(10000):
            v = generator.get_vertex()
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])
        assert np.mean(xs) == pytest.approx(0, abs=50)
        assert np.std(xs) == pytest.approx(5000/np.sqrt(12), rel=0.01)
        assert np.min(xs) >= -2500
        assert np.max(xs) <= 2500
        assert np.mean(ys) == pytest.approx(0, abs=50)
        assert np.std(ys) == pytest.approx(5000/np.sqrt(12), rel=0.01)
        assert np.min(ys) >= -2500
        assert np.max(ys) <= 2500
        assert np.mean(zs) == pytest.approx(-1500, abs=30)
        assert np.std(zs) == pytest.approx(3000/np.sqrt(12), rel=0.01)
        assert np.min(zs) >= -3000
        assert np.max(zs) <= 0

    def test_get_direction(self, generator):
        """Test that get_direction method uniformly samples on the unit sphere"""
        xs = []
        ys = []
        zs = []
        v = generator.get_direction()
        assert isinstance(v, np.ndarray)
        assert len(v) == 3
        assert np.linalg.norm(v) == pytest.approx(1)
        for _ in range(10000):
            v = generator.get_direction()
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])
        assert np.mean(xs) == pytest.approx(0, abs=0.01)
        assert np.std(xs) == pytest.approx(2/np.sqrt(12), rel=0.01)
        assert np.min(xs) >= -1
        assert np.max(xs) <= 1
        assert np.mean(ys) == pytest.approx(0, abs=0.01)
        assert np.std(ys) == pytest.approx(2/np.sqrt(12), rel=0.01)
        assert np.min(ys) >= -1
        assert np.max(ys) <= 1
        assert np.mean(zs) == pytest.approx(0, abs=0.01)
        assert np.std(zs) == pytest.approx(2/np.sqrt(12), rel=0.01)
        assert np.min(zs) >= -1
        assert np.max(zs) <= 1

    def test_get_particle_type(self, generator):
        """Test that get_particle_type method properly samples the flavor ratio"""
        flavors = []
        e_nu_nubar = []
        mu_nu_nubar = []
        tau_nu_nubar = []
        n = generator.get_particle_type()
        assert isinstance(n, Particle.Type)
        for _ in range(10000):
            n = generator.get_particle_type()
            if np.abs(n.value)==12:
                flavors.append(0)
                if n.value<0:
                    e_nu_nubar.append(0)
                else:
                    e_nu_nubar.append(1)
            elif np.abs(n.value)==14:
                flavors.append(1)
                if n.value<0:
                    mu_nu_nubar.append(0)
                else:
                    mu_nu_nubar.append(1)
            elif np.abs(n.value)==16:
                flavors.append(2)
                if n.value<0:
                    tau_nu_nubar.append(0)
                else:
                    tau_nu_nubar.append(1)
            else:
                raise ValueError("Uknown particle type thrown")
        assert np.mean(flavors) == pytest.approx(1, abs=0.02)
        assert np.std(flavors) == pytest.approx(np.sqrt(8/12), rel=0.01)
        assert np.mean(e_nu_nubar) == pytest.approx(0.78, abs=0.02)
        assert np.mean(mu_nu_nubar) == pytest.approx(0.61, abs=0.02)
        assert np.mean(tau_nu_nubar) == pytest.approx(0.61, abs=0.02)

    def test_get_exit_points(self, generator):
        """Test that the get_exit_points method returns accurate entry and exit"""
        p = Particle(particle_id=Particle.Type.electron_neutrino,
                     vertex=(0, 0, -100), direction=(1, 0, 0), energy=1e9)
        enter_point, exit_point = generator.get_exit_points(p)
        assert np.array_equal(enter_point, [-2500, 0, -100])
        assert np.array_equal(exit_point, [2500, 0, -100])

        p = Particle(particle_id=Particle.Type.electron_neutrino,
                     vertex=(0, 0, -100), direction=(0, 1, 0), energy=1e9)
        enter_point, exit_point = generator.get_exit_points(p)
        assert np.array_equal(enter_point, [0, -2500, -100])
        assert np.array_equal(exit_point, [0, 2500, -100])

        p = Particle(particle_id=Particle.Type.electron_neutrino,
                     vertex=(0, 0, -100), direction=(0, 0, 1), energy=1e9)
        enter_point, exit_point = generator.get_exit_points(p)
        assert np.array_equal(enter_point, [0, 0, -3000])
        assert np.array_equal(exit_point, [0, 0, 0])

        p = Particle(particle_id=Particle.Type.electron_neutrino,
                     vertex=(0, 0, -100), direction=(1, 1, 1), energy=1e9)
        enter_point, exit_point = generator.get_exit_points(p)
        assert np.array_equal(enter_point, [-2500, -2500, -2600])
        assert np.array_equal(exit_point, [100, 100, 0])



@pytest.fixture
def event():
    """Fixture for forming basic Event object"""
    return Event(Particle(particle_id=Particle.Type.electron_neutrino,
                          vertex=[100, 200, -500], direction=[0, 0, 1],
                          energy=1e9))

class TestListGenerator:
    """Tests for ListGenerator class"""
    def test_creation(self, event):
        """Test initialization of ListGenerator"""
        generator = ListGenerator(event)
        assert generator.events[0] == event
        assert generator.loop
        generator = ListGenerator([event, event])
        assert generator.events[0] == event
        assert generator.events[1] == event
        assert generator.loop

    def test_create_event(self, event):
        event2 = Event(Particle(particle_id="nu_e", vertex=[0, 0, 0],
                                direction=[0, 0, -1], energy=1e9))
        generator = ListGenerator([event, event2])
        assert generator.create_event() == event
        assert generator.create_event() == event2

    def test_loop(self, event):
        """Test that the loop property allows for turning on and off the
        re-iteration of the list of events"""
        event2 = Event(Particle(particle_id="nu_e", vertex=[0, 0, 0],
                                direction=[0, 0, -1], energy=1e9))
        generator = ListGenerator([event, event2])
        assert generator.create_event() == event
        assert generator.create_event() == event2
        assert generator.create_event() == event
        assert generator.create_event() == event2
        assert generator.create_event() == event
        generator = ListGenerator(event, loop=False)
        assert not generator.loop
        assert generator.create_event() == event
        with pytest.raises(StopIteration):
            generator.create_event()



test_ids = [12, -12, 14, 16]
test_vertices = [(0, 0, 0), (0, 0, -100), (-100, -100, -300), (100, 200, -500)]
test_directions = [(0, 0, -1), (0, 0, 1), (1, 0, 1), (0, 0, 1)]
test_energies = [1e9]*4
test_interactions = ["cc", "nc", "cc", "nc"]
test_weights = [0.2, 0.3, 0.4, 0.5]

@pytest.fixture
def file_gen(tmpdir):
    """Fixture for forming basic FileGenerator object,
    including creating temporary .npz files (once per test)."""
    if not "test_particles_1.npz" in [f.basename for f in tmpdir.listdir()]:
        np.savez(str(tmpdir.join("test_particles_1.npz")),
                 particle_ids=test_ids[:2], vertices=test_vertices[:2],
                 directions=test_directions[:2], energies=test_energies[:2],
                 interactions=test_interactions[:2], weights=test_weights[:2])
        np.savez(str(tmpdir.join("test_particles_2.npz")),
                 test_ids[2:], test_vertices[2:], test_directions[2:],
                 test_energies[2:], test_interactions[2:], test_weights[2:])
    return FileGenerator([str(tmpdir.join("test_particles_1.npz")),
                          str(tmpdir.join("test_particles_2.npz"))])

class TestFileGenerator:
    """Tests for FileGenerator class"""
    def test_creation(self, file_gen, tmpdir):
        """Test initialization of FileGenerator"""
        assert file_gen.files == [str(tmpdir.join("test_particles_1.npz")),
                                  str(tmpdir.join("test_particles_2.npz"))]
        assert issubclass(file_gen.interaction_model, NeutrinoInteraction)
        file_gen_2 = FileGenerator(str(tmpdir.join("test_particles_1.npz")))
        assert file_gen_2.files == [str(tmpdir.join("test_particles_1.npz"))]

    def test_create_event(self, file_gen, tmpdir):
        """Test that create_event method loops over files correctly.
        Also tests ability to read files without explicit labels since
        test_particles_2.npz is created without explicit labels"""
        for i in range(4):
            event = file_gen.create_event()
            particle = event.roots[0]
            expected = Particle(particle_id=test_ids[i],
                                vertex=test_vertices[i],
                                direction=test_directions[i],
                                energy=test_energies[i],
                                interaction_type=test_interactions[i],
                                weight=test_weights[i])
            assert particle.id == expected.id
            assert np.array_equal(particle.vertex, expected.vertex)
            assert np.array_equal(particle.direction, expected.direction)
            assert particle.energy == expected.energy
            assert particle.interaction.kind == expected.interaction.kind
            assert particle.weight == expected.weight
        with pytest.raises(StopIteration):
            file_gen.create_event()

    def test_bad_files(self, tmpdir):
        """Test that appropriate errors are raised when bad files are passed"""
        np.savez(str(tmpdir.join("bad_particles_1.npz")),
                 some=[(0, 0, 0), (0, 0, -100)], badly=[(0, 0, -1), (0, 0, 1)],
                 named=[0]*2, keys=[1e9]*2)
        with pytest.raises(KeyError):
            FileGenerator(str(tmpdir.join("bad_particles_1.npz")))
        np.savez(str(tmpdir.join("bad_particles_2.npz")),
                 [(0, 0, 0), (0, 0, -100)], [(0, 0, -1), (0, 0, 1)], [0], [1e9])
        with pytest.raises(ValueError):
            FileGenerator(str(tmpdir.join("bad_particles_2.npz")))
