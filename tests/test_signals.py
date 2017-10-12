"""File containing tests of pyrex signals module"""

import pytest

from pyrex.signals import Signal, EmptySignal, FunctionSignal

import numpy as np



@pytest.fixture
def signal():
    """Fixture for forming basic Signal object"""
    return Signal([0,1,2,3,4], [1,2,1,2,1])

@pytest.fixture(params=[([0,1,2,3,4],[0,1,2,1,0]),
                        ([0,0.1,0.2,0.3,0.4],[0,0.1,0.2,0.1,0]),
                        ([0,1e-9,2e-9,3e-9,4e-9],[0,1e-9,2e-9,1e-9,0])])
def signals(request):
    """Fixture for forming parameterized Signal objects for many value types"""
    ts = request.param[0]
    vs = request.param[1]
    return Signal(ts, vs)


class TestSignal:
    """Tests for Signal class"""
    def test_creation(self, signal):
        """Test initialization of times and values of signal"""
        assert np.array_equal(signal.times, [0,1,2,3,4])
        assert np.array_equal(signal.values, [1,2,1,2,1])

    def test_addition(self, signals):
        """Test that signal objects can be added"""
        expected = Signal(signals.times, 2*signals.values)
        signal_sum = signals + signals
        assert np.array_equal(signal_sum.times, expected.times)
        assert np.array_equal(signal_sum.values, expected.values)

    def test_signal_summation(self, signals):
        """Test that sum() can be used with signals"""
        signal_sum = sum([signals, signals])
        assert np.array_equal(signals.times, signal_sum.times)
        assert np.array_equal(signals.values+signals.values, signal_sum.values)

    def test_uniqueness(self, signals):
        """Test that a new signal made from the values of the old one are not connected"""
        new = Signal(signals.times, signals.values)
        new.times[0] = -10000
        new.values[0] = 10000
        assert new.times[0] != signals.times[0]
        assert new.values[0] != signals.values[0]

    def test_dt(self, signals):
        """Test the value of dt"""
        assert signals.dt == pytest.approx(signals.times[-1]-signals.times[-2])

    def test_dt_is_none(self):
        """Test that dt is None when not enough values are provided"""
        ts = [0]
        vs = [1]
        signal = Signal(ts, vs)
        assert signal.dt is None

    def test_dt_not_writable(self, signal):
        """Test that dt cannot be assigned to"""
        with pytest.raises(AttributeError):
            signal.dt = 0.1

    def test_envelope(self, signal):
        """Test the envelope calculation"""
        expected = [1.25653708, 2.00527169, 1, 2.00527169, 1.25653708]
        for i in range(5):
            assert signal.envelope[i] == pytest.approx(expected[i])

    def test_enveolope_not_writable(self, signal):
        """Test that envelope cannot be assigned to"""
        with pytest.raises(AttributeError):
            signal.envelope = [1,2,1,2,1]
    
    def test_resample(self, signal):
        """Test signal resampling"""
        expected = Signal(signal.times[::2], [1.2,1.6258408572364818,1.3741591427635182])
        signal.resample(3)
        for i in range(3):
            assert signal.times[i] == expected.times[i]
            assert signal.values[i] == pytest.approx(expected.values[i])


def test_empty_signal():
    """Test that an empty signal truly is empty"""
    ts = [0,1,2,3,4]
    signal = EmptySignal(ts)
    for i in range(5):
        assert signal.values[i] == 0


@pytest.mark.parametrize("func", [lambda x: x==1, np.cos])
def test_function_signal(func):
    """Test that function signal works appropriately"""
    ts = [0,1,2,3,4]
    signal = FunctionSignal(ts, func)
    for i in range(5):
        assert signal.values[i] == pytest.approx(func(ts[i]))
