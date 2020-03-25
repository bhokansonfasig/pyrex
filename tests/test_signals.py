"""File containing tests of pyrex signals module"""

import pytest

from config import SEED

from pyrex.signals import (Signal, EmptySignal, FunctionSignal,
                           GaussianNoise, ThermalNoise)

import numpy as np



@pytest.fixture
def signal():
    """Fixture for forming basic Signal object"""
    return Signal([0,1,2,3,4], [1,2,1,2,1])

@pytest.fixture(params=[([0,1,2,3,4], [0,1,2,1,0], Signal.Type.undefined),
                        ([0,0.1,0.2,0.3,0.4], [0,0.1,0.2,0.1,0],
                         Signal.Type.voltage),
                        ([0,1e-9,2e-9,3e-9,4e-9], [0,1e-9,2e-9,1e-9,0],
                         Signal.Type.field)])
def signals(request):
    """Fixture for forming parameterized Signal objects for many value types"""
    ts = request.param[0]
    vs = request.param[1]
    v_type = request.param[2]
    return Signal(ts, vs, value_type=v_type)


class TestSignal:
    """Tests for Signal class"""
    def test_creation(self, signal):
        """Test initialization of signal"""
        assert np.array_equal(signal.times, [0,1,2,3,4])
        assert np.array_equal(signal.values, [1,2,1,2,1])
        assert signal.value_type == Signal.Type.undefined

    def test_uniqueness(self, signals):
        """Test that a new signal made from the values of the old one are not connected"""
        new = Signal(signals.times, signals.values)
        new.times[0] = -10000
        new.values[0] = 10000
        assert new.times[0] != signals.times[0]
        assert new.values[0] != signals.values[0]

    def test_copy(self, signals):
        """Test that a copy of the signal is independent"""
        new = signals.copy()
        assert new is not signals
        assert np.array_equal(new.times, signals.times)
        assert new.times is not signals.times
        assert np.array_equal(new.values, signals.values)
        assert new.values is not signals.values
        assert new.value_type == signals.value_type

    def test_addition(self, signals):
        """Test that signal objects can be added"""
        expected = Signal(signals.times, 2*signals.values, signals.value_type)
        signal_sum = signals + signals
        assert np.array_equal(signal_sum.times, expected.times)
        assert np.array_equal(signal_sum.values, expected.values)
        assert signal_sum.value_type == expected.value_type

    def test_summation(self, signals):
        """Test that sum() can be used with signals"""
        signal_sum = sum([signals, signals])
        assert np.array_equal(signals.times, signal_sum.times)
        assert np.array_equal(signals.values+signals.values, signal_sum.values)
        assert signal_sum.value_type == signals.value_type

    def test_addition_type_failure(self, signal):
        """Test that signal objects cannot be added to other types
        (except left-add of iteger 0 for use of sum())"""
        assert 0+signal == signal
        with pytest.raises(TypeError):
            signal_sum = signal + 1
        with pytest.raises(TypeError):
            signal_sim = signal + [1,1,1,1,1]
        with pytest.raises(TypeError):
            signal_sum = signal + np.array([1,1,1,1,1])
        with pytest.raises(TypeError):
            signal_sim = signal + "Why would you even try this?"

    def test_addition_times_failure(self):
        """Test that signal objects with different times cannot be added"""
        signal_1 = Signal([0,1,2,3,4], [1,2,1,2,1])
        signal_2 = Signal([1,2,3,4,5], [2,3,2,3,2])
        with pytest.raises(ValueError):
            signal_sum = signal_1 + signal_2

    def test_addition_value_type_failure(self):
        """Test that adding signal objects with different value types fails"""
        signal_1 = Signal([0,1,2,3,4], [1,2,1,2,1],
                          value_type=Signal.Type.voltage)
        signal_2 = Signal([0,1,2,3,4], [2,3,2,3,2],
                          value_type=Signal.Type.field)
        with pytest.raises(ValueError):
            signal_sum = signal_1 + signal_2

    def test_addition_value_type_coercion(self, signals):
        """Test that adding signal objects with undefined value type
        results in a sum with the value type of the other signal"""
        undef_signal = Signal(signals.times, signals.values,
                              value_type=Signal.Type.undefined)
        signal_sum_1 = signals + undef_signal
        signal_sum_2 = undef_signal + signals
        assert signal_sum_1.value_type == signals.value_type
        assert signal_sum_2.value_type == signals.value_type

    @pytest.mark.parametrize("factor", [0, 2, np.pi])
    def test_multiplication(self, signals, factor):
        """Test that signal types can be multiplied by scalar values"""
        expected = Signal(signals.times, factor*signals.values, signals.value_type)
        signal_product_1 = signals * factor
        assert np.array_equal(signal_product_1.times, expected.times)
        assert np.array_equal(signal_product_1.values, expected.values)
        assert signal_product_1.value_type == expected.value_type
        signal_product_2 = factor * signals
        assert np.array_equal(signal_product_2.times, expected.times)
        assert np.array_equal(signal_product_2.values, expected.values)
        assert signal_product_2.value_type == expected.value_type
        signals *= factor
        assert np.array_equal(signals.times, expected.times)
        assert np.array_equal(signals.values, expected.values)
        assert signals.value_type == expected.value_type

    def test_multiplication_uniqueness(self, signal):
        """Test that multiplying a signal doesn't change other signals which
        used those same values"""
        signal_product = signal * 2
        signal *= 2
        assert np.array_equal(signal_product.values, signal.values)

    @pytest.mark.parametrize("factor", [2, np.pi])
    def test_division(self, signals, factor):
        """Test that signal types can be divided by scalar values"""
        expected = Signal(signals.times, signals.values/factor, signals.value_type)
        signal_quotient = signals / factor
        assert np.array_equal(signal_quotient.times, expected.times)
        assert np.array_equal(signal_quotient.values, expected.values)
        assert signal_quotient.value_type == expected.value_type
        signals /= factor
        assert np.array_equal(signals.times, expected.times)
        assert np.array_equal(signals.values, expected.values)
        assert signals.value_type == expected.value_type

    def test_value_type_setter(self, signal):
        """Test setting of signal value type by various methods"""
        signal.value_type = Signal.Type.voltage
        assert signal.value_type == Signal.Type.voltage
        signal.value_type = 2
        assert signal.value_type == Signal.Type.field
        signal.value_type = "power"
        assert signal.value_type == Signal.Type.power

    def test_value_types_equivalent(self, signal):
        """Test that value types are equivalent across classes and values"""
        assert signal.Type.voltage == Signal.Type.voltage
        assert Signal.Type.voltage == EmptySignal.Type.voltage
        assert Signal.Type.undefined == Signal.Type.unknown

    def test_dt(self, signals):
        """Test the value of dt"""
        for t1, t2 in zip(signals.times[:-1], signals.times[1:]):
            assert signals.dt == pytest.approx(t2-t1)

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
        assert len(signal.times) == 3
        assert len(signal.values) == 3
        for i in range(3):
            assert signal.times[i] == expected.times[i]
            assert signal.values[i] == pytest.approx(expected.values[i])

    def test_with_times(self, signal):
        """Test that with_times method works as expected,
        interpolating and zero-padding"""
        times = np.linspace(-2, 7, 19)
        new = signal.with_times(times)
        expected = Signal(times, [0,0,0,0,1,1.5,2,1.5,1,1.5,2,1.5,1,0,0,0,0,0,0])
        assert np.array_equal(new.values, expected.values)
        assert new.value_type == signal.value_type

    def test_shift(self, signal):
        """Test that shifting the signal changes the times accordingly"""
        old_times = np.array(signal.times)
        signal.shift(5)
        assert np.array_equal(signal.times, old_times+5)

    def test_spectrum(self, signal):
        """Test that spectrum attribute returns expected values"""
        expected = [7, -0.5, -0.5, -0.5, -0.5]
        assert len(signal.spectrum) == 5
        for i in range(5):
            assert np.real(signal.spectrum[i]) == expected[i]

    def test_frequencies(self, signal):
        """Test that frequencies attribute returns expected values"""
        expected = [0, 0.2, 0.4, -0.4, -0.2]
        assert len(signal.frequencies) == 5
        for i in range(5):
            assert signal.frequencies[i] == expected[i]

    def test_filter_frequencies(self, signal):
        """Test that the filter_frequencies method returns expected values"""
        resp = lambda f: int(np.abs(f)==0.2)
        expected = Signal(signal.times, [-0.1,0.0381966,0.1236068,0.0381966,-0.1],
                          value_type=signal.value_type)
        signal.filter_frequencies(resp)
        assert np.array_equal(signal.times, expected.times)
        for i in range(5):
            assert signal.values[i] == pytest.approx(expected.values[i])
        assert signal.value_type == expected.value_type

    def test_filter_frequencies_force_real(self, signal):
        """Test that the filter_frequencies force_real option works"""
        resp = lambda f: int(f==0.2)
        copy = Signal(signal.times, signal.values)
        expected = Signal(signal.times, [-0.05,0.0190983,0.0618034,0.0190983,-0.05],
                          value_type=signal.value_type)
        copy.filter_frequencies(resp, force_real=False)
        for i in range(5):
            assert copy.values[i] == pytest.approx(expected.values[i])
        expected = Signal(signal.times, [-0.1,0.0381966,0.1236068,0.0381966,-0.1],
                          value_type=signal.value_type)
        signal.filter_frequencies(resp, force_real=True)
        for i in range(5):
            assert signal.values[i] == pytest.approx(expected.values[i])



@pytest.fixture
def empty_signal():
    """Fixture for forming basic EmptySignal object"""
    return EmptySignal([0,1,2,3,4])


class TestEmptySignal:
    """Tests for EmptySignal class"""
    def test_creation(self, empty_signal):
        """Test initialization of empty signal"""
        assert np.array_equal(empty_signal.times, [0,1,2,3,4])
        assert np.array_equal(empty_signal.values, np.zeros(5))
        assert empty_signal.value_type == Signal.Type.undefined

    def test_copy(self, empty_signal):
        """Test that a copy of the EmptySignal is independent"""
        new = empty_signal.copy()
        assert new is not empty_signal
        assert np.array_equal(new.times, empty_signal.times)
        assert new.times is not empty_signal.times
        assert np.array_equal(new.values, empty_signal.values)
        assert new.values is not empty_signal.values
        assert new.value_type == empty_signal.value_type

    def test_addition(self, signal):
        """Test that EmptySignal objects can be added to other signals"""
        empty = EmptySignal(signal.times, signal.value_type)
        signal_sum = empty + signal
        assert np.array_equal(signal_sum.times, signal.times)
        assert np.array_equal(signal_sum.values, signal.values)
        assert signal_sum.value_type == signal.value_type

    def test_summation(self, signal):
        """Test that sum() can be used with EmptySignal objects"""
        empty = EmptySignal(signal.times, signal.value_type)
        signal_sum = sum([empty, signal])
        assert np.array_equal(signal.times, signal_sum.times)
        assert np.array_equal(signal.values, signal_sum.values)
        assert signal_sum.value_type == signal.value_type

    def test_addition_type_failure(self, empty_signal):
        """Test that EmptySignal objects cannot be added to other types
        (except left-add of iteger 0 for use of sum())"""
        assert 0+empty_signal == empty_signal
        with pytest.raises(TypeError):
            signal_sum = empty_signal + 1
        with pytest.raises(TypeError):
            signal_sim = empty_signal + [1,1,1,1,1]
        with pytest.raises(TypeError):
            signal_sum = empty_signal + np.array([1,1,1,1,1])
        with pytest.raises(TypeError):
            signal_sim = empty_signal + "Why would you even try this?"

    def test_addition_times_failure(self):
        """Test that EmptySignal objects with different times cannot be added"""
        signal_1 = EmptySignal([0,1,2,3,4])
        signal_2 = EmptySignal([1,2,3,4,5])
        with pytest.raises(ValueError):
            signal_sum = signal_1 + signal_2

    def test_addition_value_type_failure(self):
        """Test that adding EmtpySignal objects with different value types fails"""
        signal_1 = EmptySignal([0,1,2,3,4], value_type=Signal.Type.voltage)
        signal_2 = EmptySignal([0,1,2,3,4], value_type=Signal.Type.field)
        with pytest.raises(ValueError):
            signal_sum = signal_1 + signal_2

    def test_addition_value_type_coercion(self, empty_signal, signal):
        """Test that adding EmptySignal objects with undefined value type
        results in a sum with the value type of the other signal"""
        empty_signal.value_type = Signal.Type.voltage
        signal_sum_1 = signal + empty_signal
        signal_sum_2 = empty_signal + signal
        assert signal_sum_1.value_type == empty_signal.value_type
        assert signal_sum_2.value_type == empty_signal.value_type

    def test_with_times(self, empty_signal):
        """Test that with_times method keeps all zeros"""
        empty_signal.value_type = Signal.Type.voltage
        new = empty_signal.with_times(np.linspace(-2, 7, 19))
        assert np.array_equal(new.values, np.zeros(19))
        assert new.value_type == empty_signal.value_type



@pytest.fixture(params=[lambda x: np.where(x==1, 1, 0), np.cos])
def function_signals(request):
    """Fixture for forming parameterized Signal objects for many value types"""
    ts = [0,1,2,3,4]
    return FunctionSignal(ts, request.param)


class TestFunctionSignal:
    """Tests for FunctionSignal class"""
    def test_creation(self, function_signals):
        """Test initialization of function signals"""
        assert np.array_equal(function_signals.times, [0,1,2,3,4])
        for i in range(5):
            assert function_signals.values[i] == function_signals._functions[0](i)
        assert function_signals.value_type == Signal.Type.undefined

    def test_copy(self, function_signals):
        """Test that a copy of the FunctionSignal is independent"""
        new = function_signals.copy()
        assert new is not function_signals
        assert np.array_equal(new.times, function_signals.times)
        assert new.times is not function_signals.times
        assert np.array_equal(new.values, function_signals.values)
        assert new.values is not function_signals.values
        assert new.value_type == function_signals.value_type

    def test_addition(self, function_signals):
        """Test that FunctionSignal objects can be added"""
        expected = Signal(function_signals.times, 2*function_signals.values,
                          function_signals.value_type)
        signal_sum = function_signals + function_signals
        assert np.array_equal(signal_sum.times, expected.times)
        assert np.array_equal(signal_sum.values, expected.values)
        assert signal_sum.value_type == expected.value_type

    def test_summation(self, function_signals):
        """Test that sum() can be used with FunctionSignal objects"""
        signal_sum = sum([function_signals, function_signals])
        assert np.array_equal(function_signals.times, signal_sum.times)
        assert np.array_equal(2*function_signals.values, signal_sum.values)
        assert signal_sum.value_type == function_signals.value_type

    def test_addition_type_failure(self, function_signals):
        """Test that FunctionSignal objects cannot be added to other types
        (except left-add of iteger 0 for use of sum())"""
        assert 0+function_signals == function_signals
        with pytest.raises(TypeError):
            signal_sum = function_signals + 1
        with pytest.raises(TypeError):
            signal_sim = function_signals + [1,1,1,1,1]
        with pytest.raises(TypeError):
            signal_sum = function_signals + np.array([1,1,1,1,1])
        with pytest.raises(TypeError):
            signal_sim = function_signals + "Why would you even try this?"

    def test_addition_value_type_failure(self, function_signals):
        """Test that adding FunctionSignal objects with different value types fails"""
        new = function_signals.copy()
        new.value_type = Signal.Type.voltage
        function_signals.value_type = Signal.Type.field
        with pytest.raises(ValueError):
            signal_sum = new + function_signals

    def test_addition_value_type_coercion(self, function_signals):
        """Test that adding FunctionSignal objects with undefined value type
        results in a sum with the value type of the other signal"""
        function_signals.value_type = Signal.Type.voltage
        undef_signal = function_signals.copy()
        undef_signal.value_type = Signal.Type.undefined
        signal_sum_1 = function_signals + undef_signal
        signal_sum_2 = undef_signal + function_signals
        assert signal_sum_1.value_type == function_signals.value_type
        assert signal_sum_2.value_type == function_signals.value_type

    @pytest.mark.parametrize("factor", [0, 2, np.pi])
    def test_multiplication(self, function_signals, factor):
        """Test that FunctionSignal types can be multiplied by scalar values"""
        signal_product_1 = function_signals * factor
        assert np.array_equal(signal_product_1.times, function_signals.times)
        assert np.array_equal(signal_product_1.values,
                              factor*function_signals.values)
        assert signal_product_1.value_type == function_signals.value_type
        signal_product_2 = factor * function_signals
        assert np.array_equal(signal_product_2.times, function_signals.times)
        assert np.array_equal(signal_product_2.values,
                              factor*function_signals.values)
        assert signal_product_2.value_type == function_signals.value_type
        new = function_signals.copy()
        new *= factor
        assert np.array_equal(new.times, function_signals.times)
        assert np.array_equal(new.values, factor*function_signals.values)
        assert new.value_type == function_signals.value_type

    def test_multiplication_uniqueness(self, function_signals):
        """Test that multiplying a FunctionSignal doesn't change other signals which
        used those same values"""
        signal_product = function_signals * 2
        function_signals *= 2
        assert np.array_equal(signal_product.values, function_signals.values)

    @pytest.mark.parametrize("factor", [2, np.pi])
    def test_division(self, function_signals, factor):
        """Test that FunctionSignal types can be divided by scalar values"""
        signal_quotient = function_signals / factor
        assert np.array_equal(signal_quotient.times, function_signals.times)
        assert np.allclose(signal_quotient.values,
                           function_signals.values/factor)
        assert signal_quotient.value_type == function_signals.value_type
        new = function_signals.copy()
        new /= factor
        assert np.array_equal(new.times, function_signals.times)
        assert np.allclose(new.values, function_signals.values/factor)
        assert new.value_type == function_signals.value_type

    def test_resample(self, function_signals):
        """Test FunctionSignal resampling"""
        resampled = function_signals.copy()
        resampled.resample(3)
        expected_values = function_signals._functions[0](
            np.asarray(function_signals.times[::2])
        )
        assert len(resampled.times) == 3
        assert len(resampled.values) == 3
        assert np.array_equal(resampled.times, function_signals.times[::2])
        assert np.array_equal(resampled.values, expected_values)

    def test_with_times(self, function_signals):
        """Test that with_times method uses function to re-evaluate"""
        times = np.linspace(-2, 7, 19)
        function_signals.value_type == Signal.Type.voltage
        new = function_signals.with_times(times)
        assert np.array_equal(new.times, times)
        for i in range(19):
            assert new.values[i] == function_signals._functions[0](times[i])
        assert new.value_type == function_signals.value_type

    def test_shift(self, function_signals):
        """Test that shifting the FunctionSignal changes the times accordingly"""
        old_times = np.array(function_signals.times)
        function_signals.shift(5)
        assert np.array_equal(function_signals.times, old_times+5)

    def test_set_buffers(self):
        """Test whether setting the FunctionSignal time buffers works as expected"""
        signal = FunctionSignal([0, 1, 2, 3, 4], np.cos)
        assert np.array_equal(signal._buffers[0], [0, 0])
        signal.set_buffers(leading=1)
        assert np.array_equal(signal._buffers[0], [1, 0])
        signal.set_buffers(trailing=2)
        assert np.array_equal(signal._buffers[0], [1, 2])
        signal.set_buffers(trailing=1)
        assert np.array_equal(signal._buffers[0], [1, 2])
        signal.set_buffers(trailing=1, force=True)
        assert np.array_equal(signal._buffers[0], [1, 1])



@pytest.fixture
def gauss_signal():
    """Gaussian noise sample"""
    np.random.seed(SEED)
    return GaussianNoise(np.linspace(0, 100, 10001), 1)


class TestGaussianNoise:
    """Tests for GaussianNoise class"""
    def test_creation(self, gauss_signal):
        """Test initialization of gaussian noise signal"""
        assert np.array_equal(gauss_signal.times, np.linspace(0, 100, 10001))
        assert gauss_signal.sigma == 1
        assert gauss_signal.value_type == Signal.Type.voltage

    def test_mean_rms(self, gauss_signal):
        """Test the mean and standard deviation of the gaussian noise signal"""
        assert np.mean(gauss_signal.values) == pytest.approx(0, abs=5e-3)
        assert (np.std(gauss_signal.values) ==
                pytest.approx(gauss_signal.sigma, rel=0.05))



@pytest.fixture
def thermal_signal():
    """Thermal noise sample"""
    np.random.seed(SEED)
    return ThermalNoise(times=np.linspace(0, 1e-6, 10001), f_band=(300e6, 700e6),
                        rms_voltage=1)


class TestThermalNoise:
    """Tests for ThermalNoise class"""
    def test_creation(self, thermal_signal):
        """Test initialization of thermal noise signal"""
        assert np.array_equal(thermal_signal.times, np.linspace(0, 1e-6, 10001))
        assert thermal_signal.value_type == Signal.Type.voltage
        assert thermal_signal.f_min == 300e6
        assert thermal_signal.f_max == 700e6
        assert len(thermal_signal.freqs) == 400
        assert thermal_signal.rms == 1

    def test_creation_failure(self):
        """Test that initialization only succeeds if temperature and resistance
        are given (or rms_voltage)"""
        with pytest.raises(ValueError):
            noise = ThermalNoise(times=[0,1,2], f_band=(0, 100))
        with pytest.raises(ValueError):
            noise = ThermalNoise(times=[0,1,2], f_band=(0, 100),
                                 temperature=200)
        with pytest.raises(ValueError):
            noise = ThermalNoise(times=[0,1,2], f_band=(0, 100),
                                 resistance=100)

    def test_rms_voltage_precedence(self):
        """Test that rms_voltage if specified takes precedence over temperature
        and resistance values"""
        noise = ThermalNoise(times=[0,1,2], f_band=(0, 100), rms_voltage=1,
                             temperature=200, resistance=100)
        assert noise.rms == 1

    def test_mean_rms(self, thermal_signal):
        """Test the mean and standard deviation of the thermal noise signal"""
        assert np.mean(thermal_signal.values) == pytest.approx(0, abs=5e-3)
        assert (np.std(thermal_signal.values) ==
                pytest.approx(thermal_signal.rms, rel=0.05))

    def test_with_times(self, thermal_signal):
        """Test that a shifted signal gives similar values in the same time range"""
        expected_values = np.interp(np.linspace(0.5e-6, 1e-6, 5001),
                                    thermal_signal.times, thermal_signal.values)
        shifted = thermal_signal.with_times(np.linspace(0.5e-6, 1.5e-6, 10001))
        assert np.allclose(shifted.values[:5001], expected_values)

    def test_resample(self, thermal_signal):
        """Test that a downsampled signal gives similar values"""
        expected_values = np.interp(np.linspace(0, 1e-6, 1001),
                                    thermal_signal.times, thermal_signal.values)
        resampled = thermal_signal.with_times(np.linspace(0, 1e-6, 1001))
        assert np.allclose(resampled.values, expected_values)
