"""File containing tests of pyrex generation module"""

import pytest

from config import SEED

from pyrex.generation import (Generator, CylindricalGenerator,
                              RectangularGenerator, ListGenerator,
                              FileGenerator)
from pyrex.particle import Event, Particle, NeutrinoInteraction
from pyrex.earth_model import earth
from pyrex.io import File

from collections import Counter
import numpy as np


@pytest.fixture
def base_generator():
    """Fixture for forming basic Generator object"""
    return Generator(energy=1e9)

class TestGenerator:
    """Tests for base Generator class"""
    def test_creation(self, base_generator):
        """Test initialization of Generator"""
        assert base_generator.get_energy() == 1e9
        assert not base_generator.shadow
        assert np.array_equal(base_generator.ratio, np.ones(3)/3)
        assert base_generator.source == Generator.SourceType.cosmogenic
        assert issubclass(base_generator.interaction_model, NeutrinoInteraction)
        assert isinstance(base_generator.earth_model, type(earth))
        assert base_generator.count == 0

    def test_energy_generator(self):
        """Test that initialization works with an energy function or float value
        and fails otherwise"""
        generator = Generator(energy=1e9)
        assert generator.get_energy() == 1e9
        generator2 = Generator(energy=lambda: 1e9)
        assert generator2.get_energy() == 1e9
        with pytest.raises(ValueError):
            generator3 = Generator(energy=[1e9])

    def test_source_assignment(self):
        """Test that the source model can be assigned as expected"""
        generator = Generator(energy=1e9, source='astrophysical')
        assert generator.source == Generator.SourceType.astrophysical
        generator.source = 'cosmogenic'
        assert generator.source == Generator.SourceType.cosmogenic
        generator.source = Generator.SourceType.pp
        assert generator.source == Generator.SourceType.astrophysical
        generator.source = Generator.SourceType.pgamma
        assert generator.source == Generator.SourceType.cosmogenic

    def test_volume(self, base_generator):
        """Test that the volume parameter is not implemented in the base
        Generator"""
        with pytest.raises(NotImplementedError):
            base_generator.volume

    def test_solid_angle(self, base_generator):
        """Test that the default solid angle is 4pi"""
        assert base_generator.solid_angle == 4*np.pi

    def test_get_vertex(self, base_generator):
        """Test that the get_vertex method is not implemented in the base
        Generator"""
        with pytest.raises(NotImplementedError):
            base_generator.get_vertex()

    def test_get_exit_points(self, base_generator):
        """Test that the get_exit_points method is not implemented in the base
        Generator"""
        with pytest.raises(NotImplementedError):
            particle = Particle('nu_e', (0, 0, -1000), (0, 0, 1), 1e9)
            base_generator.get_exit_points(particle)

    def test_get_direction(self, base_generator):
        """Test that the get_direction method returns a unit vector direction"""
        np.random.seed(SEED)
        for _ in range(10000):
            direction = base_generator.get_direction()
            assert len(direction)==3
            assert np.linalg.norm(direction) == pytest.approx(1)

    def test_direction_distribution(self, base_generator):
        """Test that get_direction method uniformly samples on the unit sphere"""
        np.random.seed(SEED)
        xs = []
        ys = []
        zs = []
        for _ in range(10000):
            direction = base_generator.get_direction()
            xs.append(direction[0])
            ys.append(direction[1])
            zs.append(direction[2])
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

    def test_get_particle_type(self, base_generator):
        """Test that the base generator returns all neutrino types"""
        np.random.seed(SEED)
        types = [base_generator.get_particle_type() for _ in range(100)]
        assert Particle.Type.electron_neutrino in types
        assert Particle.Type.electron_antineutrino in types
        assert Particle.Type.muon_neutrino in types
        assert Particle.Type.muon_antineutrino in types
        assert Particle.Type.tau_neutrino in types
        assert Particle.Type.tau_antineutrino in types

    def test_type_distribution(self):
        """Test that the get_particle_type method samples appropriately
        based on source type and flavor ratio"""
        np.random.seed(SEED)
        total = 100000
        tolerance = 0.02
        astro_generator = Generator(1e9, source='astrophysical')
        counts = Counter(astro_generator.get_particle_type()
                         for _ in range(total))
        assert (counts[Particle.Type.electron_neutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.electron_antineutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.muon_neutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.muon_antineutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.tau_neutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.tau_antineutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        astro_generator2 = Generator(1e9, source='astrophysical',
                                     flavor_ratio=(1, 2, 0))
        counts = Counter(astro_generator2.get_particle_type()
                         for _ in range(total))
        assert (counts[Particle.Type.electron_neutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.electron_antineutrino]/total ==
                pytest.approx(1/6, rel=tolerance))
        assert (counts[Particle.Type.muon_neutrino]/total ==
                pytest.approx(2/6, rel=tolerance))
        assert (counts[Particle.Type.muon_antineutrino]/total ==
                pytest.approx(2/6, rel=tolerance))
        assert counts[Particle.Type.tau_neutrino]/total == 0
        assert counts[Particle.Type.tau_antineutrino]/total == 0
        cosmo_generator = Generator(1e9, source='cosmogenic')
        counts = Counter(cosmo_generator.get_particle_type()
                         for _ in range(total))
        assert (counts[Particle.Type.electron_neutrino]/total ==
                pytest.approx(0.78/3, rel=tolerance))
        assert (counts[Particle.Type.electron_antineutrino]/total ==
                pytest.approx(0.22/3, rel=tolerance))
        assert (counts[Particle.Type.muon_neutrino]/total ==
                pytest.approx(0.61/3, rel=tolerance))
        assert (counts[Particle.Type.muon_antineutrino]/total ==
                pytest.approx(0.39/3, rel=tolerance))
        assert (counts[Particle.Type.tau_neutrino]/total ==
                pytest.approx(0.61/3, rel=tolerance))
        assert (counts[Particle.Type.tau_antineutrino]/total ==
                pytest.approx(0.39/3, rel=tolerance))
        cosmo_generator2 = Generator(1e9, source='cosmogenic',
                                     flavor_ratio=(1, 2, 0))
        counts = Counter(cosmo_generator2.get_particle_type()
                         for _ in range(total))
        assert (counts[Particle.Type.electron_neutrino]/total ==
                pytest.approx(0.78/3, rel=tolerance))
        assert (counts[Particle.Type.electron_antineutrino]/total ==
                pytest.approx(0.22/3, rel=tolerance))
        assert (counts[Particle.Type.muon_neutrino]/total ==
                pytest.approx(2*0.61/3, rel=tolerance))
        assert (counts[Particle.Type.muon_antineutrino]/total ==
                pytest.approx(2*0.39/3, rel=tolerance))
        assert counts[Particle.Type.tau_neutrino]/total == 0
        assert counts[Particle.Type.tau_antineutrino]/total == 0

    def test_get_weights(self, base_generator):
        """Test that the get_weights method raises an error since no volume is
        defined by default"""
        particle = Particle('nu_e', (0, 0, -1000), (0, 0, 1), 1e9)
        with pytest.raises(NotImplementedError):
            weights = base_generator.get_weights(particle)

    def test_create_event(self, base_generator):
        """Test that create_event method raises an error since no volume is
        defined by default"""
        with pytest.raises(NotImplementedError):
            base_generator.create_event()



@pytest.fixture
def cyl_generator():
    """Fixture for forming basic CylindricalGenerator object"""
    return CylindricalGenerator(dr=5000, dz=3000, energy=1e9)

class TestCylindricalGenerator:
    """Tests for CylindricalGenerator class"""
    def test_creation(self, cyl_generator):
        """Test initialization of CylindricalGenerator"""
        assert cyl_generator.dr == 5000
        assert cyl_generator.dz == 3000
        assert cyl_generator.get_energy() == 1e9
        assert not cyl_generator.shadow
        assert np.array_equal(cyl_generator.ratio, np.ones(3)/3)
        assert cyl_generator.source == Generator.SourceType.cosmogenic
        assert issubclass(cyl_generator.interaction_model, NeutrinoInteraction)
        assert isinstance(cyl_generator.earth_model, type(earth))
        assert cyl_generator.count == 0

    def test_volume(self):
        """Test that the volume parameter gives the expected cylinder volume"""
        for radius in np.linspace(0, 10000, 11):
            for depth in np.linspace(0, 3000, 7):
                generator = CylindricalGenerator(dr=radius, dz=depth,
                                                 energy=1e9)
                assert generator.volume == np.pi * radius**2 * depth

    def test_solid_angle(self, cyl_generator):
        """Test that the solid angle is 4pi"""
        assert cyl_generator.solid_angle == 4*np.pi

    def test_get_vertex(self, cyl_generator):
        """Test that the get_vertex method returns a vector position inside
        the volume"""
        np.random.seed(SEED)
        for _ in range(10000):
            vertex = cyl_generator.get_vertex()
            assert len(vertex)==3
            assert np.sqrt(vertex[0]**2+vertex[1]**2) <= cyl_generator.dr
            assert -cyl_generator.dz <= vertex[2] <= 0

    def test_vertex_distribution(self, cyl_generator):
        """Test that get_vertex method uniformly samples in the cylindrical
        volume"""
        np.random.seed(SEED)
        r2s = []
        zs = []
        for _ in range(10000):
            vertex = cyl_generator.get_vertex()
            r2s.append(vertex[0]**2 + vertex[1]**2)
            zs.append(vertex[2])
        assert np.mean(r2s) == pytest.approx(cyl_generator.dr**2/2, rel=0.01)
        assert np.std(r2s) == pytest.approx(cyl_generator.dr**2/np.sqrt(12), rel=0.01)
        assert np.min(r2s) >= 0
        assert np.max(r2s) <= cyl_generator.dr**2
        assert np.mean(zs) == pytest.approx(-cyl_generator.dz/2, rel=0.01)
        assert np.std(zs) == pytest.approx(cyl_generator.dz/np.sqrt(12), rel=0.01)
        assert np.min(zs) >= -3000
        assert np.max(zs) <= 0

    def test_get_exit_points(self, cyl_generator):
        """Test that the get_exit_points method returns appropriate exit points"""
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 0, 1))
        with np.errstate(divide='ignore'):
            points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, 0, -3000))
        assert np.array_equal(points[1], (0, 0, 0))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 0, -1))
        with np.errstate(divide='ignore'):
            points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, 0, 0))
        assert np.array_equal(points[1], (0, 0, -3000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(1, 0, 0))
        points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-5000, 0, -1000))
        assert np.array_equal(points[1], (5000, 0, -1000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 1, 0))
        points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, -5000, -1000))
        assert np.array_equal(points[1], (0, 5000, -1000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(1, 0, 1))
        points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-2000, 0, -3000))
        assert np.array_equal(points[1], (1000, 0, 0))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(-4000, 0, -1000), direction=(1, 0, 1))
        points = cyl_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-5000, 0, -2000))
        assert np.array_equal(points[1], (-3000, 0, 0))

        np.random.seed(SEED)
        edge2 = cyl_generator.dr**2
        for _ in range(1000):
            particle = Particle(particle_id=cyl_generator.get_particle_type(),
                                vertex=cyl_generator.get_vertex(),
                                direction=cyl_generator.get_direction(),
                                energy=cyl_generator.get_energy())
            points = cyl_generator.get_exit_points(particle)
            for point in points:
                assert (point[0]**2+point[1]**2 == pytest.approx(edge2) or
                        point[2] == pytest.approx(0) or
                        point[2] == pytest.approx(-cyl_generator.dz))

    def test_get_direction(self, cyl_generator):
        """Test that the get_direction method returns a unit vector direction"""
        np.random.seed(SEED)
        for _ in range(10000):
            direction = cyl_generator.get_direction()
            assert len(direction)==3
            assert np.linalg.norm(direction) == pytest.approx(1)

    def test_direction_distribution(self, cyl_generator):
        """Test that get_direction method uniformly samples on the unit sphere"""
        np.random.seed(SEED)
        xs = []
        ys = []
        zs = []
        for _ in range(10000):
            direction = cyl_generator.get_direction()
            xs.append(direction[0])
            ys.append(direction[1])
            zs.append(direction[2])
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

    def test_get_weights(self, cyl_generator):
        """Test that the get_weights returns floats between 0 and 1"""
        np.random.seed(SEED)
        for _ in range(1000):
            particle = Particle(particle_id=cyl_generator.get_particle_type(),
                                vertex=cyl_generator.get_vertex(),
                                direction=cyl_generator.get_direction(),
                                energy=cyl_generator.get_energy())
            weights = cyl_generator.get_weights(particle)
            assert len(weights)==2
            assert 0 <= weights[0] <= 1
            assert 0 <= weights[1] <= 1

    def test_weights_distribution(self, cyl_generator):
        """Test that the get_weights distributions are as expected"""
        np.random.seed(SEED)
        survival_weights = []
        interaction_weights = []
        for _ in range(10000):
            particle = Particle(particle_id=cyl_generator.get_particle_type(),
                                vertex=cyl_generator.get_vertex(),
                                direction=cyl_generator.get_direction(),
                                energy=cyl_generator.get_energy())
            weights = cyl_generator.get_weights(particle)
            survival_weights.append(weights[0])
            interaction_weights.append(weights[1])
        survival_hist, _ = np.histogram(survival_weights, 
                                        range=(0, 1), bins=20)
        expected = np.array([4537.79, 74.1, 49.14, 43.61, 36.18, 29.1, 24.88,
                             20.71, 17.09, 14.46, 12.01, 9.93, 7.43, 4.97,
                             12.87, 27.77, 36.09, 47.87, 92.14, 4901.86])
        assert np.allclose(survival_hist, expected, rtol=0.05, atol=10)

        # TODO: Add test of interaction weight distribution

    def test_create_event(self, cyl_generator):
        """Test that create_event method returns an Event object"""
        event = cyl_generator.create_event()
        assert isinstance(event, Event)

    def test_count(self, cyl_generator):
        """Test that the count increments appropriately when throwing events"""
        np.random.seed(SEED)
        assert cyl_generator.count == 0
        for i in range(100):
            cyl_generator.create_event()
            assert cyl_generator.count == i+1
        shadow_generator = CylindricalGenerator(dr=5000, dz=3000, energy=1e9,
                                                shadow=True)
        assert shadow_generator.count == 0
        counts = [shadow_generator.count]
        for i in range(500):
            shadow_generator.create_event()
            counts.append(shadow_generator.count)
        assert np.all(np.diff(counts)>=1)
        assert np.max(np.diff(counts)>1)
        assert np.mean(np.diff(counts)) == pytest.approx(2, rel=0.05)

    def test_shadow_weights(self):
        """Test that the get_weights returns the appropriate weights when in
        earth shadowing mode"""
        np.random.seed(SEED)
        generator = CylindricalGenerator(dr=5000, dz=3000, energy=1e9,
                                         shadow=True)
        for _ in range(1000):
            particle = generator.create_event().roots[0]
            assert particle.survival_weight == 1
            assert 0 <= particle.interaction_weight <= 1



@pytest.fixture
def rect_generator():
    """Fixture for forming basic RectangularGenerator object"""
    return RectangularGenerator(dx=5000, dy=5000, dz=3000, energy=1e9)

class TestRectangularGenerator:
    """Tests for RectangularGenerator class"""
    def test_creation(self, rect_generator):
        """Test initialization of RectangularGenerator"""
        assert rect_generator.dx == 5000
        assert rect_generator.dy == 5000
        assert rect_generator.dz == 3000
        assert rect_generator.get_energy() == 1e9
        assert not rect_generator.shadow
        assert np.array_equal(rect_generator.ratio, np.ones(3)/3)
        assert rect_generator.source == Generator.SourceType.cosmogenic
        assert issubclass(rect_generator.interaction_model, NeutrinoInteraction)
        assert isinstance(rect_generator.earth_model, type(earth))
        assert rect_generator.count == 0

    def test_volume(self):
        """Test that the volume parameter gives the expected box volume"""
        for width in np.linspace(0, 10000, 11):
            for length in np.linspace(0, 10000, 11):
                for depth in np.linspace(0, 3000, 7):
                    generator = RectangularGenerator(dx=width, dy=length,
                                                      dz=depth, energy=1e9)
                    assert generator.volume == width * length * depth

    def test_solid_angle(self, rect_generator):
        """Test that the solid angle is 4pi"""
        assert rect_generator.solid_angle == 4*np.pi

    def test_get_vertex(self, rect_generator):
        """Test that the get_vertex method returns a vector position inside
        the volume"""
        np.random.seed(SEED)
        for _ in range(10000):
            vertex = rect_generator.get_vertex()
            assert len(vertex)==3
            assert -rect_generator.dx/2 <= vertex[0] <= rect_generator.dx/2
            assert -rect_generator.dy/2 <= vertex[1] <= rect_generator.dy/2
            assert -rect_generator.dz <= vertex[2] <= 0

    def test_vertex_distribution(self, rect_generator):
        """Test that get_vertex method uniformly samples in the box volume"""
        np.random.seed(SEED)
        xs = []
        ys = []
        zs = []
        for _ in range(10000):
            vertex = rect_generator.get_vertex()
            xs.append(vertex[0])
            ys.append(vertex[1])
            zs.append(vertex[2])
        assert np.mean(xs) == pytest.approx(0, abs=50)
        assert np.std(xs) == pytest.approx(rect_generator.dx/np.sqrt(12), rel=0.01)
        assert np.min(xs) >= -rect_generator.dx/2
        assert np.max(xs) <= rect_generator.dx/2
        assert np.mean(ys) == pytest.approx(0, abs=50)
        assert np.std(ys) == pytest.approx(rect_generator.dy/np.sqrt(12), rel=0.01)
        assert np.min(ys) >= -rect_generator.dy/2
        assert np.max(ys) <= rect_generator.dy/2
        assert np.mean(zs) == pytest.approx(-rect_generator.dz/2, rel=0.01)
        assert np.std(zs) == pytest.approx(rect_generator.dz/np.sqrt(12), rel=0.01)
        assert np.min(zs) >= -3000
        assert np.max(zs) <= 0

    def test_get_exit_points(self, rect_generator):
        """Test that the get_exit_points method returns appropriate exit points"""
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 0, 1))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, 0, -3000))
        assert np.array_equal(points[1], (0, 0, 0))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 0, -1))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, 0, 0))
        assert np.array_equal(points[1], (0, 0, -3000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(1, 0, 0))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-2500, 0, -1000))
        assert np.array_equal(points[1], (2500, 0, -1000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(0, 1, 0))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (0, -2500, -1000))
        assert np.array_equal(points[1], (0, 2500, -1000))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(0, 0, -1000), direction=(1, 0, 1))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-2000, 0, -3000))
        assert np.array_equal(points[1], (1000, 0, 0))
        particle = Particle(particle_id='nu_e', energy=1e9,
                            vertex=(-2000, 0, -1000), direction=(1, 0, 1))
        points = rect_generator.get_exit_points(particle)
        assert np.array_equal(points[0], (-2500, 0, -1500))
        assert np.array_equal(points[1], (-1000, 0, 0))

        np.random.seed(SEED)
        for _ in range(1000):
            particle = Particle(particle_id=rect_generator.get_particle_type(),
                                vertex=rect_generator.get_vertex(),
                                direction=rect_generator.get_direction(),
                                energy=rect_generator.get_energy())
            points = rect_generator.get_exit_points(particle)
            for point in points:
                assert (point[0] == pytest.approx(rect_generator.dx/2) or
                        point[0] == pytest.approx(-rect_generator.dx/2) or
                        point[1] == pytest.approx(rect_generator.dy/2) or
                        point[1] == pytest.approx(-rect_generator.dy/2) or
                        point[2] == pytest.approx(0) or
                        point[2] == pytest.approx(-rect_generator.dz))

    def test_get_direction(self, rect_generator):
        """Test that the get_direction method returns a unit vector direction"""
        np.random.seed(SEED)
        for _ in range(10000):
            direction = rect_generator.get_direction()
            assert len(direction)==3
            assert np.linalg.norm(direction) == pytest.approx(1)

    def test_direction_distribution(self, rect_generator):
        """Test that get_direction method uniformly samples on the unit sphere"""
        np.random.seed(SEED)
        xs = []
        ys = []
        zs = []
        for _ in range(10000):
            direction = rect_generator.get_direction()
            xs.append(direction[0])
            ys.append(direction[1])
            zs.append(direction[2])
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

    def test_get_weights(self, rect_generator):
        """Test that the get_weights returns floats between 0 and 1"""
        np.random.seed(SEED)
        for _ in range(1000):
            particle = Particle(particle_id=rect_generator.get_particle_type(),
                                vertex=rect_generator.get_vertex(),
                                direction=rect_generator.get_direction(),
                                energy=rect_generator.get_energy())
            weights = rect_generator.get_weights(particle)
            assert len(weights)==2
            assert 0 <= weights[0] <= 1
            assert 0 <= weights[1] <= 1

    def test_weights_distribution(self, rect_generator):
        """Test that the get_weights distributions are as expected"""
        np.random.seed(SEED)
        survival_weights = []
        interaction_weights = []
        for _ in range(10000):
            particle = Particle(particle_id=rect_generator.get_particle_type(),
                                vertex=rect_generator.get_vertex(),
                                direction=rect_generator.get_direction(),
                                energy=rect_generator.get_energy())
            weights = rect_generator.get_weights(particle)
            survival_weights.append(weights[0])
            interaction_weights.append(weights[1])
        survival_hist, _ = np.histogram(survival_weights, 
                                        range=(0, 1), bins=20)
        expected = np.array([4537.79, 74.1, 49.14, 43.61, 36.18, 29.1, 24.88,
                             20.71, 17.09, 14.46, 12.01, 9.93, 7.43, 4.97,
                             12.87, 27.77, 36.09, 47.87, 92.14, 4901.86])
        assert np.allclose(survival_hist, expected, rtol=0.05, atol=10)

        # TODO: Add test of interaction weight distribution

    def test_create_event(self, rect_generator):
        """Test that create_event method returns an Event object"""
        event = rect_generator.create_event()
        assert isinstance(event, Event)

    def test_count(self, rect_generator):
        """Test that the count increments appropriately when throwing events"""
        np.random.seed(SEED)
        assert rect_generator.count == 0
        for i in range(100):
            rect_generator.create_event()
            assert rect_generator.count == i+1
        shadow_generator = RectangularGenerator(dx=5000, dy=5000, dz=3000,
                                                energy=1e9, shadow=True)
        assert shadow_generator.count == 0
        counts = [shadow_generator.count]
        for i in range(500):
            shadow_generator.create_event()
            counts.append(shadow_generator.count)
        assert np.all(np.diff(counts)>=1)
        assert np.max(np.diff(counts)>1)
        assert np.mean(np.diff(counts)) == pytest.approx(2, rel=0.05)

    def test_shadow_weights(self):
        """Test that the get_weights returns the appropriate weights when in
        earth shadowing mode"""
        np.random.seed(SEED)
        generator = RectangularGenerator(dx=5000, dy=5000, dz=3000, energy=1e9,
                                         shadow=True)
        for _ in range(1000):
            particle = generator.create_event().roots[0]
            assert particle.survival_weight == 1
            assert 0 <= particle.interaction_weight <= 1



@pytest.fixture
def events():
    """Fixture for forming basic list of Event objects"""
    return [
        Event(Particle(particle_id=Particle.Type.electron_neutrino,
                       vertex=[100, 200, -500], direction=[0, 0, 1],
                       energy=1e9)),
        Event(Particle(particle_id=Particle.Type.electron_antineutrino,
                       vertex=[0, 0, 0], direction=[0, 0, -1],
                       energy=1e9)),
    ]

class TestListGenerator:
    """Tests for ListGenerator class"""
    def test_creation(self, events):
        """Test initialization of ListGenerator with various argument types"""
        generator = ListGenerator(events)
        assert generator.events[0] == events[0]
        assert generator.events[1] == events[1]
        assert generator.loop
        assert generator.count == 0
        generator = ListGenerator(events[0])
        assert generator.events[0] == events[0]
        assert generator.loop
        assert generator.count == 0
        generator = ListGenerator(events[0].roots[0])
        assert generator.events[0].roots[0] == events[0].roots[0]
        assert generator.loop
        assert generator.count == 0

    def test_create_event(self, events):
        """Test that create_event returns the appropriate given events"""
        generator = ListGenerator(events)
        assert generator.create_event() == events[0]
        assert generator.create_event() == events[1]

    def test_loop(self, events):
        """Test that the loop property allows for turning on and off the
        re-iteration of the list of events"""
        generator = ListGenerator(events)
        assert generator.create_event() == events[0]
        assert generator.create_event() == events[1]
        assert generator.create_event() == events[0]
        assert generator.create_event() == events[1]
        assert generator.create_event() == events[0]
        generator = ListGenerator(events, loop=False)
        assert not generator.loop
        for i, match_event in enumerate(events):
            assert generator.create_event() == match_event
        with pytest.raises(StopIteration):
            generator.create_event()

    def test_count(self, events):
        """Test that the count increments appropriately when throwing events"""
        generator = ListGenerator(events)
        assert generator.count == 0
        for i in range(5):
            generator.create_event()
            assert generator.count == i+1
        generator = ListGenerator(events, loop=False)
        assert generator.count == 0
        for i in range(len(events)):
            generator.create_event()
            assert generator.count == i+1
        with pytest.raises(StopIteration):
            generator.create_event()
        assert generator.count == 2



@pytest.fixture
def file_generator(tmpdir):
    """Fixture for forming basic FileGenerator object, including creating a
    temporary output file to be read from (once per test)."""
    if not "test_output_1.h5" in [f.basename for f in tmpdir.listdir()]:
        writer1 = File(str(tmpdir.join('test_output_1.h5')), mode='w',
                       write_particles=True, write_triggers=False,
                       write_antenna_triggers=False, write_rays=False,
                       write_noise=False, write_waveforms=False,
                       require_trigger=False)
        writer2 = File(str(tmpdir.join('test_output_2.h5')), mode='w',
                       write_particles=True, write_triggers=False,
                       write_antenna_triggers=False, write_rays=False,
                       write_noise=False, write_waveforms=False,
                       require_trigger=False)
        writer1.open()
        writer2.open()
        np.random.seed(SEED)
        gen = CylindricalGenerator(dr=5000, dz=3000, energy=1e9)
        for _ in range(10):
            writer1.add(gen.create_event())
        for _ in range(10):
            writer2.add(gen.create_event())
        writer1.close()
        writer2.close()
    return FileGenerator([str(tmpdir.join('test_output_1.h5')),
                          str(tmpdir.join('test_output_2.h5'))])

@pytest.fixture
def shadow_file_generator(tmpdir):
    """Fixture for forming FileGenerator object from a shadowed generator,
    including creating a temporary output file to be read from (once per test)."""
    if not "test_shadow_output.h5" in [f.basename for f in tmpdir.listdir()]:
        writer = File(str(tmpdir.join('test_shadow_output.h5')), mode='w',
                      write_particles=True, write_triggers=False,
                      write_antenna_triggers=False, write_rays=False,
                      write_noise=False, write_waveforms=False,
                      require_trigger=False)
        writer.open()
        np.random.seed(SEED)
        gen = CylindricalGenerator(dr=5000, dz=3000, energy=1e9, shadow=True)
        for _ in range(10):
            prev = gen.count
            event = gen.create_event()
            writer.add(event, events_thrown=gen.count-prev)
        writer.close()
    return FileGenerator(str(tmpdir.join('test_shadow_output.h5')))

class TestFileGenerator:
    """Tests for FileGenerator class"""
    def test_creation(self, file_generator, tmpdir):
        """Test initialization of FileGenerator"""
        assert file_generator.files == [str(tmpdir.join("test_output_1.h5")),
                                        str(tmpdir.join("test_output_2.h5"))]
        assert issubclass(file_generator.interaction_model, NeutrinoInteraction)
        assert file_generator.slice_range == 100
        assert file_generator.count == 0
        file_generator2 = FileGenerator(str(tmpdir.join("test_output_1.h5")),
                                        slice_range=1)
        assert file_generator2.files == [str(tmpdir.join("test_output_1.h5"))]
        assert file_generator2.slice_range == 1
        assert file_generator2.count == 0

    def test_create_event(self, file_generator, tmpdir):
        """Test that create_event method loops over files correctly"""
        np.random.seed(SEED)
        gen = CylindricalGenerator(dr=5000, dz=3000, energy=1e9)
        expected_events = [gen.create_event() for _ in range(20)]
        for i in range(20):
            event = file_generator.create_event()
            particle = event.roots[0]
            expected = expected_events[i].roots[0]
            assert particle.id == expected.id
            assert np.array_equal(particle.vertex, expected.vertex)
            assert np.array_equal(particle.direction, expected.direction)
            assert particle.energy == expected.energy
            assert particle.interaction.kind == expected.interaction.kind
            assert particle.weight == expected.weight
        with pytest.raises(StopIteration):
            file_generator.create_event()

    def test_slice_range(self, file_generator, tmpdir):
        """Test that the slice range is obeyed when iterating files"""
        assert (len(file_generator._events) ==
                min(file_generator.slice_range, 10))
        file_generator2 = FileGenerator(str(tmpdir.join("test_output_1.h5")),
                                        slice_range=1)
        assert (len(file_generator2._events) ==
                min(file_generator2.slice_range, 10))

    def test_count(self, file_generator, shadow_file_generator, tmpdir):
        """Test that the count increments appropriately when throwing events,
        event for shadow-based generators"""
        assert file_generator.count == 0
        for i in range(20):
            file_generator.create_event()
            assert file_generator.count == i+1
        assert shadow_file_generator.count == 0
        counts = [shadow_file_generator.count]
        for i in range(10):
            shadow_file_generator.create_event()
            counts.append(shadow_file_generator.count)
        assert np.all(np.diff(counts)>=1)
        assert np.max(np.diff(counts)) > 1
