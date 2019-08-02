"""File containing tests of pyrex internal_functions module"""

import pytest

from config import SEED

from pyrex.internal_functions import (normalize, get_from_enum,
                                      flatten, mirror_func,
                                      lazy_property, LazyMutableClass)

from enum import Enum
import inspect
import types
import numpy as np



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
