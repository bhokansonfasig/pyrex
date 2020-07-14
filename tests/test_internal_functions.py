"""File containing tests of pyrex internal_functions module"""

import pytest

from config import SEED

from pyrex.internal_functions import (normalize, get_from_enum,
                                      flatten, complex_interp,
                                      complex_bilinear_interp, mirror_func,
                                      lazy_property, LazyMutableClass)

from enum import Enum
import inspect
import types
import numpy as np
import scipy.signal



class Test_normalize:
    """Tests for normalize function"""
    def test_normalization(self):
        """Test that vectors are successfully normalized"""
        np.random.seed(SEED)
        for _ in range(1000):
            vector = np.random.normal(size=3)
            if np.array_equal(vector, [0, 0, 0]):
                continue
            unit = normalize(vector)
            quotient = vector/unit
            assert np.linalg.norm(unit) == pytest.approx(1)
            assert quotient[0] == pytest.approx(quotient[1])
            assert quotient[0] == pytest.approx(quotient[2])

    def test_zero(self):
        """Test that the zero vector doesn't cause problems"""
        unit = normalize([0, 0, 0])
        assert np.array_equal(unit, [0, 0, 0])



@pytest.fixture
def enum():
    """Fixture for forming basic enum"""
    class Color(Enum):
        red = 1
        rouge = 1
        green = 2
        vert = 2
        blue = 3
        bleu = 3
    return Color


class Test_get_from_enum:
    """Tests for get_from_enum function"""
    def test_get_value(self, enum):
        """Test that getting value directly from value works"""
        assert get_from_enum(enum.red, enum) == enum.red
        assert get_from_enum(enum.green, enum) == enum.green
        assert get_from_enum(enum.blue, enum) == enum.blue

    def test_get_int(self, enum):
        """Test that getting vlue from integer works"""
        assert get_from_enum(1, enum) == enum.red
        assert get_from_enum(2, enum) == enum.green
        assert get_from_enum(3, enum) == enum.blue

    def test_get_str(self, enum):
        """Test that getting value from name works"""
        assert get_from_enum("red", enum) == enum.red
        assert get_from_enum("green", enum) == enum.green
        assert get_from_enum("blue", enum) == enum.blue

    def test_get_ambiguous(self, enum):
        """Test that getting ambiguous values from any method works"""
        assert get_from_enum(enum.rouge, enum) == enum.red
        assert get_from_enum(2, enum) == enum.vert
        assert get_from_enum("bleu", enum) == enum.blue



class Test_flatten:
    """Tests for flatten function"""
    def test_generator(self):
        """Test that the returned object is a generator object"""
        l = [1, 2, 3]
        assert isinstance(flatten(l), types.GeneratorType)

    def test_flat_list(self):
        """Test that a flat list isn't changed"""
        l = [1, 2, 3]
        assert np.array_equal(list(flatten(l)), l)

    def test_empty_list(self):
        """Test that flattening an empty list doesn't cause trouble"""
        assert np.array_equal(list(flatten([])), [])

    def test_single_nest(self):
        """Test that a single nested layer can be flattened"""
        l1 = [1, 2, [3, 4]]
        assert np.array_equal(list(flatten(l1)), [1, 2, 3, 4])
        l2 = [1, [2, 3], 4, [5, 6, 7], 8]
        assert np.array_equal(list(flatten(l2)), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_double_nest(self):
        """Test that a double nested layer can be flattened"""
        l1 = [1, 2, [3, [4, 5]]]
        assert np.array_equal(list(flatten(l1)), [1, 2, 3, 4, 5])
        l2 = [1, [2, 3], 4, [5, [6, 7], 8]]
        assert np.array_equal(list(flatten(l2)), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_varied_iterables(self):
        """Test that multiple iterable types can be flattened at once"""
        l1 = [1, 2, (3, 4), [5, 6]]
        assert np.array_equal(list(flatten(l1)), [1, 2, 3, 4, 5, 6])
        l2 = [1, [2, (3, 4), 5, [6, 7], 8]]
        assert np.array_equal(list(flatten(l2)), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_flatten_flattened(self):
        """Test that list containing flattened list (i.e. generator objects)
        can be flattened"""
        l = [1, 2, [3, flatten([4, [5, 6]])]]
        assert np.array_equal(list(flatten(l)), [1, 2, 3, 4, 5, 6])

    def test_strings_nonrecursive(self):
        """Test that strings don't cause a recursive explosion
        (by just not flattening them)"""
        l = [1, "two", 3]
        assert np.array_equal(list(flatten(l)), [1, "two", 3])

    def test_dont_flatten(self):
        """Test that dont_flatten argument is obeyed"""
        l1 = [1, 2, (3, 4)]
        f1 = flatten(l1, dont_flatten=[tuple])
        assert list(f1) == l1
        l2 = [1, 2, (3, [4, 5]), 6]
        f2 = flatten(l2, dont_flatten=[tuple])
        assert list(f2) == l2
        l3 = [1, 2, (3, [4, 5]), 6, [7, 8]]
        f3 = flatten(l3, dont_flatten=[tuple])
        assert list(f3) == [1, 2, (3, [4, 5]), 6, 7, 8]



class Test_complex_interp:
    """Tests for complex_interp function"""
    def test_cartesian_interpolation(self):
        """Test cartesian complex interpolation"""
        steps = np.arange(5)
        interp_steps = steps[:-1] + 0.5
        real = np.array([1, 2, 3, 2, 1])
        imag = np.array([3, 2, 1, 2, 3])
        interp = complex_interp(interp_steps, steps, real+1j*imag,
                                method='cartesian')
        real_interp = np.interp(interp_steps, steps, real)
        imag_interp = np.interp(interp_steps, steps, imag)
        assert np.array_equal(np.real(interp), real_interp)
        assert np.array_equal(np.imag(interp), imag_interp)

    def test_cartesian_extrapolation(self):
        """Test cartesian complex extrapolation"""
        steps = np.arange(5)
        extrap_steps = [steps[0]-1, steps[-1]+1]
        real = np.array([1, 2, 3, 2, 1])
        imag = np.array([3, 2, 1, 2, 3])
        extrap = complex_interp(extrap_steps, steps, real+1j*imag,
                                method='cartesian', outer=None)
        assert np.array_equal(np.real(extrap), [real[0], real[-1]])
        assert np.array_equal(np.imag(extrap), [imag[0], imag[-1]])
        extrap2 = complex_interp(extrap_steps, steps, real+imag,
                                 method='cartesian', outer=0)
        assert np.array_equal(extrap2, [0, 0])

    def test_euler_interpolation(self):
        """Test euler complex interpolation"""
        steps = np.arange(5)
        interp_steps = steps[:-1] + 0.5
        gain = np.array([1, 2, 3, 2, 1])
        phase = np.radians([45, 90, 180, 90, 45])
        interp = complex_interp(interp_steps, steps, gain*np.exp(1j*phase),
                                method='euler')
        gain_interp = np.interp(interp_steps, steps, gain)
        phase_interp = np.interp(interp_steps, steps, phase)
        assert np.allclose(np.abs(interp), gain_interp)
        assert np.array_equal(np.angle(interp), phase_interp)

    def test_euler_extrapolation(self):
        """Test euler complex extrapolation"""
        steps = np.arange(5)
        extrap_steps = [steps[0]-1, steps[-1]+1]
        gain = np.array([1, 2, 3, 2, 1])
        phase = np.radians([45, 90, 180, 90, 45])
        extrap = complex_interp(extrap_steps, steps, gain*np.exp(1j*phase),
                                method='euler', outer=None)
        assert np.allclose(np.abs(extrap), [gain[0], gain[-1]])
        assert np.allclose(np.angle(extrap), [phase[0], phase[-1]])
        extrap2 = complex_interp(extrap_steps, steps, gain*np.exp(1j*phase),
                                 method='euler', outer=0)
        assert np.array_equal(extrap2, [0, 0])

    def test_euler_phase_unwrapping(self):
        """Test phase unwrapping in euler complex interpolation"""
        steps = np.arange(5)
        interp_steps = steps[:-1] + 0.5
        gain = np.array([1, 2, 3, 2, 1])
        phase = np.radians([45, 90, 180, -90, -45])
        interp = complex_interp(interp_steps, steps, gain*np.exp(1j*phase),
                                method='euler')
        phase_interp = np.interp(interp_steps, steps, np.unwrap(phase))
        phase_interp = phase_interp%(2*np.pi)
        phase_interp[phase_interp>np.pi] -= 2*np.pi
        assert np.allclose(np.angle(interp), phase_interp)



class Test_complex_bilinear_interp:
    """Tests for complex_bilinear_interp function"""
    def test_cartesian_interpolation(self):
        """Test cartesian complex bilinear interpolation"""
        x_steps = np.arange(5)
        y_steps = np.arange(4)
        x_interp_steps = x_steps[:-1] + 0.5
        y_interp_steps = y_steps[:-1] + 0.5
        real = np.array([[[1, 2, 3, 2],
                          [2, 4, 6, 4],
                          [3, 6, 9, 6],
                          [2, 3, 4, 3],
                          [1, 2, 3, 2]]])
        imag = np.array([[[3, 2, 1, 2],
                          [4, 3, 2, 3],
                          [5, 4, 3, 4],
                          [4, 3, 3, 3],
                          [3, 2, 1, 2]]])
        average_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
        real_interp = scipy.signal.convolve2d(real[0], average_kernel,
                                              mode='valid')
        imag_interp = scipy.signal.convolve2d(imag[0], average_kernel,
                                              mode='valid')
        for i, x in enumerate(x_interp_steps):
            for j, y in enumerate(y_interp_steps):
                interp = complex_bilinear_interp(x, y, x_steps, y_steps,
                                                 real+1j*imag,
                                                 method='cartesian')
                assert np.real(interp)==real_interp[i, j]
                assert np.imag(interp)==imag_interp[i, j]

    def test_euler_interpolation(self):
        """Test euler complex bilinear interpolation"""
        x_steps = np.arange(5)
        y_steps = np.arange(4)
        x_interp_steps = x_steps[:-1] + 0.5
        y_interp_steps = y_steps[:-1] + 0.5
        gain = np.array([[[1, 2, 3, 2],
                          [2, 4, 6, 4],
                          [3, 6, 9, 6],
                          [2, 3, 4, 3],
                          [1, 2, 3, 2]]])
        phase = np.radians([[[90, 45, 0, 45],
                             [135, 90, 45, 90],
                             [180, 135, 90, 45],
                             [135, 90, 45, 90],
                             [90, 45, 0, 45]]])
        average_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
        gain_interp = scipy.signal.convolve2d(gain[0], average_kernel,
                                              mode='valid')
        phase_interp = scipy.signal.convolve2d(phase[0], average_kernel,
                                               mode='valid')
        for i, x in enumerate(x_interp_steps):
            for j, y in enumerate(y_interp_steps):
                interp = complex_bilinear_interp(x, y, x_steps, y_steps,
                                                 gain*np.exp(1j*phase),
                                                 method='euler')
                assert np.abs(interp)==pytest.approx(gain_interp[i, j])
                assert np.angle(interp)==pytest.approx(phase_interp[i, j])

    def test_euler_phase_unwrapping(self):
        """Test phase unwrapping in euler complex bilinear interpolation"""
        x_steps = np.arange(5)
        y_steps = np.arange(4)
        x_interp_steps = x_steps[:-1] + 0.5
        y_interp_steps = y_steps[:-1] + 0.5
        gain = np.array([[[1, 2, 3, 2],
                          [2, 4, 6, 4],
                          [3, 6, 9, 6],
                          [2, 3, 4, 3],
                          [1, 2, 3, 2]],
                         [[1, 2, 3, 2],
                          [2, 4, 6, 4],
                          [3, 6, 9, 6],
                          [2, 3, 4, 3],
                          [1, 2, 3, 2]]])
        phase = np.radians([[[90, 45, 0, 45],
                             [135, 90, 45, 90],
                             [180, 135, 90, 45],
                             [135, 90, 45, 90],
                             [90, 45, 0, 45]],
                            [[90, 45, 0, 45],
                             [45, 0, -45, 0],
                             [0, -45, -90, -135],
                             [-135, 180, 135, 180],
                             [90, 45, 0, 45]]])
        unwrapped = np.unwrap(phase, axis=0)
        average_kernel = np.array([[0.25, 0.25], [0.25, 0.25]])
        gain_interp_0 = scipy.signal.convolve2d(gain[0], average_kernel,
                                                mode='valid')
        gain_interp_1 = scipy.signal.convolve2d(gain[1], average_kernel,
                                                mode='valid')
        phase_interp_0 = scipy.signal.convolve2d(unwrapped[0], average_kernel,
                                                 mode='valid')
        phase_interp_1 = scipy.signal.convolve2d(unwrapped[1], average_kernel,
                                                 mode='valid')
        for i, x in enumerate(x_interp_steps):
            for j, y in enumerate(y_interp_steps):
                interp = complex_bilinear_interp(x, y, x_steps, y_steps,
                                                 gain*np.exp(1j*phase),
                                                 method='euler',
                                                 unwrap_axis=0)
                assert np.allclose(np.abs(interp), [gain_interp_0[i, j],
                                                    gain_interp_1[i, j]])
                assert np.allclose(np.angle(interp), [phase_interp_0[i, j],
                                                      phase_interp_1[i, j]])



@pytest.fixture
def mirror():
    """Fixture for forming basic mirrored function"""
    def run_func(*args):
        return sum(args)
    def match_func(a, b):
        """Sums two values"""
        return 0
    return mirror_func(match_func, run_func)


class Test_mirror_func:
    """Tests for mirror_func function"""
    def test_keeps_functionality(self, mirror):
        """Test that the mirrored function matches the run function's
        functionality"""
        assert mirror(1, 2) == 3

    def test_matches_docstring(self, mirror):
        """Test that the mirroring function matches the match function's
        docstring"""
        assert mirror.__doc__ == "Sums two values"

    def test_matches_arguments(self, mirror):
        """Test that the mirrored function matches the match function's
        argument signature and not the run function's argument signature"""
        signature = inspect.signature(mirror)
        assert 'a' in signature.parameters
        assert 'b' in signature.parameters
        assert 'args' not in signature.parameters



class MyClass:
    def __init__(self, a):
        self.a = a

    @lazy_property
    def b(self):
        return (self.a, self.a)


class Test_lazy_property:
    """Tests for lazy_property decorator"""
    def test_is_property(self):
        """Test that a lazy property works as a property"""
        lazy = MyClass(1)
        assert isinstance(lazy.b, tuple)
        assert lazy.b == (1, 1)
        with pytest.raises(AttributeError):
            lazy.b = (2, 2)

    def test_is_lazy(self):
        """Test that the lazy property is lazy and therefore not re-evaluated"""
        lazy = MyClass(1)
        assert lazy.b is lazy._lazy_b
        assert lazy.b is lazy.b

    def test_lazy_badness(self):
        """Test that lazy property is too strong and will not be re-evaluated
        if the class properties change"""
        lazy = MyClass(1)
        assert lazy.b == (1, 1)
        lazy.a = 2
        assert lazy.b != (2, 2)



class MyLazyClass(LazyMutableClass):
    def __init__(self, a):
        self.a = a
        super().__init__()

    @lazy_property
    def b(self):
        return (self.a, self.a)

class SlightlyLazyClass(LazyMutableClass):
    def __init__(self, a, c):
        self.a = a
        self.c = c
        super().__init__(static_attributes=['a'])

    @lazy_property
    def b(self):
        return (self.a, self.a)

class TestLazyMutableClass:
    """Tests for LazyMutableClass class"""
    def test_creation(self):
        """Test initialization of LazyMutableClass"""
        lazy1 = LazyMutableClass()
        assert lazy1._static_attrs == []
        lazy2 = MyLazyClass(1)
        assert lazy2._static_attrs == ['a']
        lazy3 = SlightlyLazyClass(1, 2)
        assert lazy3._static_attrs == ['a']

    def test_lazy_property(self):
        """Test that a lazy property is truly lazy"""
        lazy = SlightlyLazyClass(1, 2)
        assert lazy.b is lazy._lazy_b
        assert lazy.b is lazy.b

    def test_lazy_reset(self):
        """Test that a lazy property is reset if one of the specified
        static_attrs changes"""
        lazy = SlightlyLazyClass(1, 2)
        assert lazy.b == (1, 1)
        lazy.c = 5
        assert lazy.b == (1, 1)
        lazy.a = 2
        assert lazy.b == (2, 2)
