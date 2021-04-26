"""File containing tests of pyrex ray_tracing module"""

import pytest

from pyrex.ray_tracing import (BasicRayTracer, BasicRayTracePath,
                               SpecializedRayTracer, SpecializedRayTracePath)
from pyrex.ice_model import AntarcticIce, ice

import numpy as np



@pytest.fixture
def ray_tracer():
    """Fixture for forming basic SpecializedRayTracer object"""
    return SpecializedRayTracer(from_point=[100, 200, -500],
                                to_point=[0, 0, -100],
                                ice_model=ice)

@pytest.fixture
def ray_tracer2():
    """Fixture for forming basic SpecializedRayTracer object"""
    return SpecializedRayTracer(from_point=[100, 1000, -200],
                                to_point=[-100, -100, -300],
                                ice_model=ice)

@pytest.fixture
def ray_tracer3():
    """Fixture for forming basic SpecializedRayTracer object"""
    return SpecializedRayTracer(from_point=[0, 0, -1500],
                                to_point=[-100, -100, -200],
                                ice_model=ice)

@pytest.fixture
def bad_tracer():
    """Fixture for forming SpecializedRayTracer object with no solutions"""
    return SpecializedRayTracer(from_point=[500, 500, -100],
                                to_point=[0, 0, -100],
                                ice_model=ice)

@pytest.fixture
def out_tracer():
    """Fixture for forming SpecializedRayTracer object with point above ice"""
    return SpecializedRayTracer(from_point=[100, 200, -500],
                                to_point=[0, 0, 100],
                                ice_model=ice)


class TestSpecializedRayTracer:
    """Tests for SpecializedRayTracer class"""
    def test_creation(self, ray_tracer):
        """Test initialization of ray_tracer"""
        assert isinstance(ray_tracer, BasicRayTracer)
        assert np.array_equal(ray_tracer.from_point, [100, 200, -500])
        assert np.array_equal(ray_tracer.to_point, [0, 0, -100])
        assert ray_tracer.ice == ice
        assert ray_tracer.dz == 1
        assert issubclass(ray_tracer.solution_class, BasicRayTracePath)
        assert ray_tracer._static_attrs == ['from_point', 'to_point',
                                            'ice', 'dz']

    def test_properties(self, ray_tracer):
        """Test basic properties of ray_tracer"""
        assert ray_tracer.z_turn_proximity == 1 / 10
        assert ray_tracer.z0 == -500
        assert ray_tracer.z1 == -100
        assert ray_tracer.n0 == ice.index(-500)
        assert ray_tracer.rho == np.sqrt(100**2 + 200**2)

    def test_special_properties(self, ray_tracer):
        """Test properties of ray_tracer specific to SpecializedRayTracer"""
        assert isinstance(ray_tracer, SpecializedRayTracer)
        assert issubclass(ray_tracer.solution_class, SpecializedRayTracePath)
        assert ray_tracer.valid_ice_model
        assert ray_tracer.z_uniform < -500

    def test_max_angle(self, ray_tracer, ray_tracer2, ray_tracer3):
        """Test the maximum launch angle between the ray_tracer points"""
        expected = np.arcsin(ice.index(-100)/ice.index(-500))
        assert ray_tracer.max_angle == expected
        expected2 = np.arcsin(ice.index(-200)/ice.index(-300))
        assert ray_tracer2.max_angle == expected2
        expected3 = np.arcsin(ice.index(-200)/ice.index(-1500))
        assert ray_tracer3.max_angle == expected3

    def test_peak_angle(self, ray_tracer, ray_tracer2, ray_tracer3,
                        rel=1e-6):
        """Test the angle at which the indirect solutions curve peaks"""
        # TODO: Add peak_angle tests for more from/to_points
        expected = 1.189505626
        assert ray_tracer.peak_angle == pytest.approx(expected, rel=rel)
        expected2 = 1.335645314
        assert ray_tracer2.peak_angle == pytest.approx(expected2, rel=rel)
        expected3 = 1.378722855
        assert ray_tracer3.peak_angle == pytest.approx(expected3, rel=rel)

    def test_direct_r_max(self, ray_tracer, ray_tracer2, ray_tracer3,
                          rel=1e-6):
        """Test the maximum r-value of a direct path for ray_tracer"""
        # TODO: Add direct_r_max tests for more from/to_points
        expected = 1343.040443
        assert ray_tracer.direct_r_max == pytest.approx(expected, rel=rel)
        expected2 = 1032.691176
        assert ray_tracer2.direct_r_max == pytest.approx(expected2, rel=rel)
        expected3 = 7472.020234
        assert ray_tracer3.direct_r_max == pytest.approx(expected3, rel=rel)

    def test_indirect_r_max(self, ray_tracer, ray_tracer2, ray_tracer3,
                            rel=1e-6):
        """Test the maximum r-value of an indirect path for ray_tracer"""
        # TODO: Add indirect_r_max tests for more from/to_points
        expected = 1413.629124
        assert ray_tracer.indirect_r_max == pytest.approx(expected, rel=rel)
        expected2 = 1440.711968
        assert ray_tracer2.indirect_r_max == pytest.approx(expected2, rel=rel)
        expected3 = 7453.552318
        assert ray_tracer3.indirect_r_max == pytest.approx(expected3, rel=rel)

    def test_expected_solutions(self, ray_tracer, ray_tracer2, ray_tracer3,
                                bad_tracer, out_tracer):
        """Test expected solutions for good and bad ray tracers"""
        assert ray_tracer.expected_solutions == [True, False, True]
        assert ray_tracer2.expected_solutions == [False, True, True]
        assert ray_tracer3.expected_solutions == [True, False, True]
        assert bad_tracer.expected_solutions == [False, False, False]
        assert out_tracer.expected_solutions == [False, False, False]

    def test_exists(self, ray_tracer, ray_tracer2, ray_tracer3,
                    bad_tracer, out_tracer):
        """Test solution existence for good and bad ray tracers"""
        assert ray_tracer.exists
        assert ray_tracer2.exists
        assert ray_tracer3.exists
        assert not bad_tracer.exists
        assert not out_tracer.exists

    def test_direct_angle(self, ray_tracer, ray_tracer2, ray_tracer3,
                          rel=1e-6):
        """Test direct solution angle for ray_tracer"""
        # TODO: Add direct_angle tests for more from/to_points
        expected = 0.502922872
        assert ray_tracer.direct_angle == pytest.approx(expected, rel=rel)
        assert ray_tracer2.direct_angle is None
        expected3 = 0.108249386
        assert ray_tracer3.direct_angle == pytest.approx(expected3, rel=rel)

    def test_indirect_angle_1(self, ray_tracer, ray_tracer2, ray_tracer3,
                              rel=1e-6):
        """Test first indirect solution angle for ray_tracer"""
        # TODO: Add indirect_angle_1 tests for more from/to_points
        assert ray_tracer.indirect_angle_1 is None
        expected2 = 1.550728133
        assert ray_tracer2.indirect_angle_1 == pytest.approx(expected2, rel=rel)
        assert ray_tracer3.indirect_angle_1 is None

    def test_indirect_angle_2(self, ray_tracer, ray_tracer2, ray_tracer3,
                              rel=1e-6):
        """Test second indirect solution angle for ray_tracer"""
        # TODO: Add indirect_angle_2 tests for more from/to_points
        expected = 0.334937507
        assert ray_tracer.indirect_angle_2 == pytest.approx(expected, rel=rel)
        expected2 = 1.105377817
        assert ray_tracer2.indirect_angle_2 == pytest.approx(expected2, rel=rel)
        expected3 = 0.081055943
        assert ray_tracer3.indirect_angle_2 == pytest.approx(expected3, rel=rel)

    def test_solutions(self, ray_tracer, ray_tracer2, ray_tracer3,
                       bad_tracer, out_tracer):
        """Test solutions types and lengths for good and bad ray tracers"""
        assert len(ray_tracer.solutions) == 2
        for sol in ray_tracer.solutions:
            assert isinstance(sol, BasicRayTracePath)
        assert len(ray_tracer2.solutions) == 2
        for sol in ray_tracer2.solutions:
            assert isinstance(sol, BasicRayTracePath)
        assert len(ray_tracer3.solutions) == 2
        for sol in ray_tracer3.solutions:
            assert isinstance(sol, BasicRayTracePath)
        assert len(bad_tracer.solutions) == 0
        assert len(out_tracer.solutions) == 0



@pytest.fixture
def basic_ray_tracer():
    """Fixture for forming basic SpecializedRayTracer object"""
    return BasicRayTracer(from_point=[100, 200, -500],
                          to_point=[0, 0, -100],
                          ice_model=ice)

@pytest.fixture
def basic_ray_tracer2():
    """Fixture for forming basic SpecializedRayTracer object"""
    return BasicRayTracer(from_point=[100, 1000, -200],
                          to_point=[-100, -100, -300],
                          ice_model=ice)

@pytest.fixture
def basic_ray_tracer3():
    """Fixture for forming basic SpecializedRayTracer object"""
    return BasicRayTracer(from_point=[0, 0, -1500],
                          to_point=[-100, -100, -200],
                          ice_model=ice)

@pytest.fixture
def basic_bad_tracer():
    """Fixture for forming SpecializedRayTracer object with no solutions"""
    return BasicRayTracer(from_point=[500, 500, -100],
                          to_point=[0, 0, -100],
                          ice_model=ice)

@pytest.fixture
def basic_out_tracer():
    """Fixture for forming SpecializedRayTracer object with point above ice"""
    return BasicRayTracer(from_point=[100, 200, -500],
                          to_point=[0, 0, 100],
                          ice_model=ice)


class TestBasicRayTracer(TestSpecializedRayTracer):
    """Tests for BasicRayTracer class"""
    def test_creation(self, basic_ray_tracer):
        """Test initialization of ray_tracer"""
        super().test_creation(basic_ray_tracer)

    def test_properties(self, basic_ray_tracer):
        """Test basic properties of ray_tracer"""
        super().test_creation(basic_ray_tracer)

    def test_max_angle(self, basic_ray_tracer, basic_ray_tracer2,
                       basic_ray_tracer3):
        """Test the maximum launch angle between the ray_tracer points"""
        super().test_max_angle(basic_ray_tracer, basic_ray_tracer2,
                               basic_ray_tracer3)

    def test_peak_angle(self, basic_ray_tracer, basic_ray_tracer2,
                        basic_ray_tracer3):
        """Test the angle at which the indirect solutions curve peaks"""
        super().test_peak_angle(basic_ray_tracer, basic_ray_tracer2,
                                basic_ray_tracer3, rel=0.01)

    def test_direct_r_max(self, basic_ray_tracer, basic_ray_tracer2,
                          basic_ray_tracer3):
        """Test the maximum r-value of a direct path for ray_tracer"""
        super().test_direct_r_max(basic_ray_tracer, basic_ray_tracer2,
                                  basic_ray_tracer3, rel=0.01)

    def test_indirect_r_max(self, basic_ray_tracer, basic_ray_tracer2,
                            basic_ray_tracer3):
        """Test the maximum r-value of an indirect path for ray_tracer"""
        super().test_indirect_r_max(basic_ray_tracer, basic_ray_tracer2,
                                    basic_ray_tracer3, rel=0.01)

    def test_expected_solutions(self, basic_ray_tracer, basic_ray_tracer2,
                                basic_ray_tracer3, basic_bad_tracer,
                                basic_out_tracer):
        """Test expected solutions for good and bad ray tracers"""
        super().test_expected_solutions(basic_ray_tracer, basic_ray_tracer2,
                                        basic_ray_tracer3, basic_bad_tracer,
                                        basic_out_tracer)

    def test_exists(self, basic_ray_tracer, basic_ray_tracer2,
                    basic_ray_tracer3, basic_bad_tracer, basic_out_tracer):
        """Test solution existence for good and bad ray tracers"""
        super().test_exists(basic_ray_tracer, basic_ray_tracer2,
                            basic_ray_tracer3, basic_bad_tracer,
                            basic_out_tracer)

    def test_direct_angle(self, basic_ray_tracer, basic_ray_tracer2,
                          basic_ray_tracer3):
        """Test direct solution angle for ray_tracer"""
        super().test_direct_angle(basic_ray_tracer, basic_ray_tracer2,
                                  basic_ray_tracer3, rel=0.01)

    def test_indirect_angle_1(self, basic_ray_tracer, basic_ray_tracer2,
                              basic_ray_tracer3):
        """Test first indirect solution angle for ray_tracer"""
        super().test_indirect_angle_1(basic_ray_tracer, basic_ray_tracer2,
                                      basic_ray_tracer3, rel=0.01)

    def test_indirect_angle_2(self, basic_ray_tracer, basic_ray_tracer2,
                              basic_ray_tracer3):
        """Test second indirect solution angle for ray_tracer"""
        super().test_indirect_angle_2(basic_ray_tracer, basic_ray_tracer2,
                                      basic_ray_tracer3, rel=0.01)

    def test_solutions(self, basic_ray_tracer, basic_ray_tracer2,
                       basic_ray_tracer3, basic_bad_tracer, basic_out_tracer):
        """Test solutions types and lengths for good and bad ray tracers"""
        super().test_solutions(basic_ray_tracer, basic_ray_tracer2,
                               basic_ray_tracer3, basic_bad_tracer,
                               basic_out_tracer)



class TestSpecializedRayTracePath:
    """Tests for SpecializedRayTracePath class"""
    def test_creation(self, ray_tracer):
        """Test initialization of paths from ray_tracer"""
        path_1 = ray_tracer.solutions[0]
        path_2 = ray_tracer.solutions[1]
        assert np.array_equal(path_1.from_point, ray_tracer.from_point)
        assert np.array_equal(path_1.to_point, ray_tracer.to_point)
        assert path_1.theta0 == ray_tracer.direct_angle
        assert path_1.ice == ray_tracer.ice
        assert path_1.dz == ray_tracer.dz
        assert path_1.direct
        assert np.array_equal(path_2.from_point, ray_tracer.from_point)
        assert np.array_equal(path_2.to_point, ray_tracer.to_point)
        assert path_2.theta0 == ray_tracer.indirect_angle_2
        assert path_2.ice == ray_tracer.ice
        assert path_2.dz == ray_tracer.dz
        assert not path_2.direct

    def test_properties(self, ray_tracer):
        """Test basic properties of paths from ray_tracer"""
        path_1 = ray_tracer.solutions[0]
        path_2 = ray_tracer.solutions[1]
        assert path_1.z_turn_proximity == ray_tracer.z_turn_proximity
        assert path_1.z0 == -500
        assert path_1.z1 == -100
        assert path_1.n0 == ice.index(-500)
        assert path_1.rho == ray_tracer.rho
        assert path_1.phi == np.arctan2(-200, -100)
        assert path_2.z_turn_proximity == ray_tracer.z_turn_proximity
        assert path_2.z0 == -500
        assert path_2.z1 == -100
        assert path_2.n0 == ice.index(-500)
        assert path_2.rho == ray_tracer.rho
        assert path_2.phi == np.arctan2(-200, -100)

    def test_special_properties(self, ray_tracer):
        """Test properties of paths from ray_tracer specific to
        SpecializedRayTracePath"""
        path_1 = ray_tracer.solutions[0]
        path_2 = ray_tracer.solutions[1]
        assert path_1.uniformity_factor == 1 - 1e-5
        assert path_1.beta_tolerance == 0.005
        assert path_1.valid_ice_model
        assert path_1.z_uniform < -500
        assert path_2.uniformity_factor == 1 - 1e-5
        assert path_2.beta_tolerance == 0.005
        assert path_2.valid_ice_model
        assert path_2.z_uniform < -500

    def test_beta(self, ray_tracer, ray_tracer2, ray_tracer3, rel=1e-6):
        """Test beta attribute of paths from ray_tracer"""
        expected = [
            0.857657674, 0.584911785,
            1.748962413, 1.563246733,
            0.192307818, 0.144121643
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            assert path.beta == pytest.approx(expected[i], rel=rel)

    def test_z_turn(self, ray_tracer, ray_tracer2, ray_tracer3, rel=1e-6):
        """Test z_turn attribute of paths from ray_tracer"""
        expected = [
            0,            0,
            -199.1353225, -51.89587289,
            0,            0
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            assert path.z_turn == pytest.approx(expected[i], rel=rel)

    def test_emitted_direction(self, ray_tracer, ray_tracer2, ray_tracer3,
                               rel=1e-6, absol=1e-12):
        """Test emitted_direction attribute of paths from ray_tracer"""
        expected = [
            [-0.215551832, -0.431103664, 0.876177516],
            [-0.147003648, -0.294007296, 0.944430854],
            [-0.178849418, -0.983671799, 0.020066846],
            [-0.159858077, -0.879219424, 0.448796835],
            [-0.076394473, -0.076394473, 0.994146754],
            [-0.057252467, -0.057252467, 0.996716765]
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            for j in range(3):
                assert (path.emitted_direction[j] ==
                        pytest.approx(expected[i][j], rel=rel, abs=absol))

    def test_received_direction(self, ray_tracer, ray_tracer2, ray_tracer3,
                                rel=1e-6, absol=1e-12):
        """Test received_direction attribute of paths from ray_tracer"""
        expected = [
            [-0.230345830, -0.460691661,  0.857148757],
            [-0.157092970, -0.314185939, -0.936274000],
            [-0.176579412, -0.971186767, -0.160049914],
            [-0.157829115, -0.868060131, -0.470703282],
            [-0.077734536, -0.077734536,  0.993938974],
            [-0.058256753, -0.058256753, -0.996600372]
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            for j in range(3):
                assert (path.received_direction[j] ==
                        pytest.approx(expected[i][j], rel=rel, abs=absol))

    def test_path_length(self, ray_tracer, ray_tracer2, ray_tracer3, rel=1e-6):
        """Test path_length attribute of paths from ray_tracer"""
        expected = [
            458.2769525, 640.5791883,
            1124.124374, 1197.740306,
            1307.670715, 1705.892277
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            assert path.path_length == pytest.approx(expected[i], rel=rel)

    def test_tof(self, ray_tracer, ray_tracer2, ray_tracer3, rel=1e-6):
        """Test tof attribute of paths from ray_tracer"""
        expected = [
            2.687681956e-6, 3.599689884e-6,
            6.593891540e-6, 6.702370564e-6,
            7.756416929e-6, 9.918147150e-6
        ]
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            print(i)
            assert path.tof == pytest.approx(expected[i], rel=rel)

    @pytest.mark.parametrize("frequency", [1e3, 1e4 ,1e5, 1e6, 1e7, 1e8, 1e9])
    def test_attenuation(self, frequency, ray_tracer, ray_tracer2, ray_tracer3,
                         rel=1e-6):
        """Test of attenuation method of paths from ray_tracer"""
        expected = {
            1e3: [0.99692576, 0.99582895,
                  0.99267936, 0.99245785,
                  0.98666489, 0.98422668],
            1e4: [0.99319069, 0.99076314,
                  0.98381875, 0.98332263,
                  0.97105187, 0.96572717],
            1e5: [0.98495202, 0.97960811,
                  0.96442748, 0.96332895,
                  0.93773784, 0.92634887],
            1e6: [0.96691269, 0.95528893,
                  0.92272900, 0.92035656,
                  0.86874756, 0.84547376],
            1e7: [0.92805250, 0.90343250,
                  0.83647913, 0.83163028,
                  0.73491319, 0.69187480],
            1e8: [0.84730443, 0.79814316,
                  0.67271197, 0.66394042,
                  0.50950321, 0.44555791],
            1e9: [0.69232598, 0.60618071,
                  0.41469964, 0.40259517,
                  0.22843533, 0.16956334]
        }
        for i, path in enumerate(ray_tracer.solutions+ray_tracer2.solutions
                                 +ray_tracer3.solutions):
            assert (path.attenuation(frequency) ==
                    pytest.approx(expected[frequency][i], rel=rel))



class TestBasicRayTracePath(TestSpecializedRayTracePath):
    """Tests for BasicRayTracePath class"""
    def test_creation(self, basic_ray_tracer):
        """Test initialization of paths from ray_tracer"""
        super().test_creation(basic_ray_tracer)

    def test_properties(self, basic_ray_tracer):
        """Test basic properties of paths from ray_tracer"""
        super().test_properties(basic_ray_tracer)

    def test_beta(self, basic_ray_tracer, basic_ray_tracer2,
                  basic_ray_tracer3):
        """Test beta attribute of paths from ray_tracer"""
        super().test_beta(basic_ray_tracer, basic_ray_tracer2,
                          basic_ray_tracer3, rel=0.01)

    def test_z_turn(self, basic_ray_tracer, basic_ray_tracer2,
                    basic_ray_tracer3):
        """Test z_turn attribute of paths from ray_tracer"""
        super().test_z_turn(basic_ray_tracer, basic_ray_tracer2,
                            basic_ray_tracer3, rel=0.01)

    def test_emitted_direction(self, basic_ray_tracer, basic_ray_tracer2,
                               basic_ray_tracer3):
        """Test emitted_direction attribute of paths from ray_tracer"""
        super().test_emitted_direction(basic_ray_tracer, basic_ray_tracer2,
                                       basic_ray_tracer3, absol=0.01)

    def test_received_direction(self, basic_ray_tracer, basic_ray_tracer2,
                                basic_ray_tracer3):
        """Test received_direction attribute of paths from ray_tracer"""
        super().test_received_direction(basic_ray_tracer, basic_ray_tracer2,
                                        basic_ray_tracer3, absol=0.01)

    def test_path_length(self, basic_ray_tracer, basic_ray_tracer2,
                         basic_ray_tracer3):
        """Test path_length attribute of paths from ray_tracer"""
        super().test_path_length(basic_ray_tracer, basic_ray_tracer2,
                                 basic_ray_tracer3, rel=0.01)

    def test_tof(self, basic_ray_tracer, basic_ray_tracer2,
                 basic_ray_tracer3):
        """Test tof attribute of paths from ray_tracer"""
        super().test_tof(basic_ray_tracer, basic_ray_tracer2,
                         basic_ray_tracer3, rel=0.01)

    @pytest.mark.parametrize("frequency", [1e3, 1e4 ,1e5, 1e6, 1e7, 1e8, 1e9])
    def test_attenuation(self, frequency, basic_ray_tracer, basic_ray_tracer2,
                         basic_ray_tracer3):
        """Test of attenuation method of paths from ray_tracer"""
        super().test_attenuation(frequency, basic_ray_tracer, basic_ray_tracer2,
                                 basic_ray_tracer3, rel=0.075)
