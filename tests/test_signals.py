"""File containing tests of pyrex signals module"""

import pytest

from config import SEED

from pyrex.signals import (Signal, EmptySignal, FunctionSignal,
                           AskaryanSignal, GaussianNoise, ThermalNoise)
from pyrex.ice_model import IceModel
from pyrex.particle import Particle

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
        assert len(signal.times) == 3
        assert len(signal.values) == 3
        for i in range(3):
            assert signal.times[i] == expected.times[i]
            assert signal.values[i] == pytest.approx(expected.values[i])

    def test_value_types_equivalent(self, signal):
        """Test that value types are equivalent across classes"""
        assert signal.Type.voltage == Signal.Type.voltage
        assert Signal.Type.voltage == EmptySignal.Type.voltage

    def test_with_times(self, signal):
        """Test that with_times method works as expected,
        interpolating and zero-padding"""
        times = np.linspace(-2, 7, 19)
        new = signal.with_times(times)
        expected = Signal(times, [0,0,0,0,1,1.5,2,1.5,1,1.5,2,1.5,1,0,0,0,0,0,0])
        assert np.array_equal(new.values, expected.values)
        assert new.value_type == signal.value_type

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

    def test_with_times(self, empty_signal):
        """Test that with_times method keeps all zeros"""
        empty_signal.value_type = Signal.Type.voltage
        new = empty_signal.with_times(np.linspace(-2, 7, 19))
        assert np.array_equal(new.values, np.zeros(19))
        assert new.value_type == empty_signal.value_type



@pytest.fixture(params=[lambda x: x==1, np.cos])
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
            assert function_signals.values[i] == function_signals.function(i)
        assert function_signals.value_type == Signal.Type.undefined

    def test_with_times(self, function_signals):
        """Test that with_times method uses function to re-evaluate"""
        times = np.linspace(-2, 7, 19)
        function_signals.value_type == Signal.Type.voltage
        new = function_signals.with_times(times)
        assert np.array_equal(new.times, times)
        for i in range(19):
            assert new.values[i] == function_signals.function(times[i])
        assert new.value_type == function_signals.value_type



@pytest.fixture
def arz_pulse():
    """Example Askaryan pulse from https://arxiv.org/pdf/1106.6283v3.pdf"""
    # Create particle to ensure shower energy is 3e9 GeV
    particle = Particle(particle_id=Particle.Type.electron_neutrino,
                        vertex=(0, 0, -1000), direction=(0, 0, 1), energy=3e9,
                        interaction_type="cc")
    particle.interaction.em_frac = 1
    particle.interaction.had_frac = 0
    n = IceModel.index(particle.vertex[2])
    cherenkov_angle = np.arcsin(np.sqrt(1 - 1/n**2))
    return AskaryanSignal(times=np.linspace(0, 3e-9, 301),
                          particle=particle,
                          viewing_angle=cherenkov_angle-np.radians(0.3),
                          t0=1e-9)


class TestAskaryanSignal:
    """Tests for AksaryanSignal class"""
    def test_arz_pulse(self, arz_pulse):
        assert arz_pulse.em_energy == 3e9
        assert arz_pulse.had_energy == 0
        assert np.array_equal(arz_pulse.times, np.linspace(0, 3e-9, 301))
        assert arz_pulse.value_type == Signal.Type.field
        # FIXME: Fix the amplitude of Askaryan pulses and use these amplitude tests
        # assert np.max(arz_pulse.values) == pytest.approx(200, rel=0.1)
        # assert np.min(arz_pulse.values) == pytest.approx(-200, rel=0.1)
        peak_to_peak_time = (arz_pulse.times[np.argmin(arz_pulse.values)] -
                             arz_pulse.times[np.argmax(arz_pulse.values)])
        assert peak_to_peak_time == pytest.approx(0.2e-9, abs=0.05e-9)

    # TODO: Add tests for vector_potential, RAC, charge_profile, and max_length methods



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
    
    def test_sigma(self, gauss_signal):
        """Test the standard deviation of the gaussian noise signal"""
        assert np.std(gauss_signal.values) == pytest.approx(gauss_signal.sigma, rel=1e-2)



@pytest.fixture
def thermal_signal():
    """Thermal noise sample"""
    np.random.seed(SEED)
    return ThermalNoise(times=np.linspace(0, 100e-9, 1001), f_band=(300e6, 700e6),
                        rms_voltage=1)


class TestThermalNoise:
    """Tests for ThermalNoise class"""
    def test_creation(self, thermal_signal):
        """Test initialization of thermal noise signal"""
        assert np.array_equal(thermal_signal.times, np.linspace(0, 100e-9, 1001))
        assert thermal_signal.value_type == Signal.Type.voltage
        assert thermal_signal.f_min == 300e6
        assert thermal_signal.f_max == 700e6
        assert len(thermal_signal.freqs) == 40
        assert np.array_equal(thermal_signal.amps, np.ones(len(thermal_signal.freqs)))
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
    
    def test_rms(self, thermal_signal):
        """Test the rms value of the thermal noise signal"""
        assert (np.sqrt(np.mean(thermal_signal.values**2)) ==
                pytest.approx(thermal_signal.rms, rel=1e-3))
