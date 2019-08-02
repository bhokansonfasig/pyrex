"""File containing tests of pyrex signals module"""

import pytest

from config import SEED

from pyrex.signals import (Signal, EmptySignal, FunctionSignal,
                           ZHSAskaryanSignal, ARVZAskaryanSignal,
                           GaussianNoise, ThermalNoise)
from pyrex.ice_model import ice
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

    def test_uniqueness(self, signals):
        """Test that a new signal made from the values of the old one are not connected"""
        new = Signal(signals.times, signals.values)
        new.times[0] = -10000
        new.values[0] = 10000
        assert new.times[0] != signals.times[0]
        assert new.values[0] != signals.values[0]

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
def zhs_pulse():
    """Example ZHS Askaryan pulse"""
    # Create particle to ensure shower energy is 3e9 GeV
    particle = Particle(particle_id=Particle.Type.electron_neutrino,
                        vertex=(0, 0, -1000), direction=(0, 0, 1), energy=3e9,
                        interaction_type="cc")
    particle.interaction.em_frac = 1
    particle.interaction.had_frac = 0
    n = ice.index(particle.vertex[2])
    cherenkov_angle = np.arcsin(np.sqrt(1 - 1/n**2))
    return ZHSAskaryanSignal(times=np.linspace(0, 3e-9, 301),
                             particle=particle,
                             viewing_angle=cherenkov_angle-np.radians(0.3),
                             viewing_distance=100,
                             ice_model=ice, t0=1e-9)


class TestZHSAskaryanSignal():
    """Tests for ZHSAskaryanSignal class"""
    def test_zhs_pulse(self, zhs_pulse):
        """Test parameters of a sample ZHS signal"""
        assert zhs_pulse.energy == 3e9
        assert np.array_equal(zhs_pulse.times, np.linspace(0, 3e-9, 301))
        assert zhs_pulse.value_type == Signal.Type.field
        assert np.max(zhs_pulse.values) == pytest.approx(14.085, rel=0.1)
        assert np.min(zhs_pulse.values) == pytest.approx(-1.392, rel=0.1)
        peak_to_peak_time = (zhs_pulse.times[np.argmin(zhs_pulse.values)] -
                             zhs_pulse.times[np.argmax(zhs_pulse.values)])
        assert np.abs(peak_to_peak_time) == pytest.approx(0.4e-9, abs=0.05e-9)



@pytest.fixture
def arz_pulse():
    """Example Askaryan pulse from https://arxiv.org/pdf/1106.6283v3.pdf"""
    # Create particle to ensure shower energy is 3e9 GeV
    particle = Particle(particle_id=Particle.Type.electron_neutrino,
                        vertex=(0, 0, -1000), direction=(0, 0, 1), energy=3e9,
                        interaction_type="cc")
    particle.interaction.em_frac = 1
    particle.interaction.had_frac = 0
    n = ice.index(particle.vertex[2])
    cherenkov_angle = np.arcsin(np.sqrt(1 - 1/n**2))
    return ARVZAskaryanSignal(times=np.linspace(0, 3e-9, 301),
                              particle=particle,
                              viewing_angle=cherenkov_angle-np.radians(0.3),
                              viewing_distance=1,
                              ice_model=ice, t0=1e-9)


class TestARVZAskaryanSignal:
    """Tests for ARVZAksaryanSignal class"""
    def test_arz_pulse(self, arz_pulse):
        """Test parameters of the example ARVZ signal"""
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

    def test_vector_potential(self, arz_pulse):
        """Test vector potential of ARVZ signal sample"""
        assert np.all(np.isclose(
            arz_pulse.vector_potential,
            np.cumsum(np.concatenate(([0], arz_pulse.values)))[:-1] * -arz_pulse.dt,
            rtol=1e-5, atol=1e-10
        ))

    @pytest.mark.parametrize("energy", [1e9, 1e10, 1e11])
    def test_RAC(self, arz_pulse, energy):
        """Test vector potential at Cherenkov angle (Fig 2 of ARVZ)"""
        times = np.linspace(-1e-9, 1.5e-9, 6)
        expectations = [3.4e-8, 1.8e-7, 9.0e-6, 3.1e-7, 7.8e-8, 3.0e-8]
        factor = 1e11 / energy
        for time, expected in zip(times, expectations):
            assert (np.abs(arz_pulse.RAC(time, energy)) * factor
                    == pytest.approx(expected, rel=0.05))

    def test_em_shower_profile(self, arz_pulse):
        """Test the electromagnetic shower profile"""
        lengths = [0.1, 2, 5, 10, 15, 20, 25]
        expectations_tev = [4.5e5, 5.2e8, 9.3e8, 1.1e7, 1.8e4, 1.3e1, 7.1e-3]
        expectations_pev = [4.2e2, 1.3e7, 7.2e8, 2.6e8, 3.3e6, 1.0e4, 1.4e1]
        expectations_eev = [4.0e-1, 1.2e5, 9.7e7, 7.6e8, 7.6e7, 1.0e6, 4.3e3]
        for i, length in enumerate(lengths):
            assert (arz_pulse.em_shower_profile(z=length, energy=1e3) * 1e6 / 1.602e-19
                    == pytest.approx(expectations_tev[i], rel=0.05))
            assert (arz_pulse.em_shower_profile(z=length, energy=1e6) * 1e3 / 1.602e-19
                    == pytest.approx(expectations_pev[i], rel=0.05))
            assert (arz_pulse.em_shower_profile(z=length, energy=1e9) / 1.602e-19
                    == pytest.approx(expectations_eev[i], rel=0.05))
        lengths = np.linspace(1, 15, 1000)
        for energy in [1e3, 1e6, 1e9]:
            assert (np.max(arz_pulse.em_shower_profile(lengths, energy)) / 1.602e-19
                    == pytest.approx(energy, rel=0.3))

    def test_had_shower_profile(self, arz_pulse):
        """Test the hadronic shower profile"""
        lengths = [0.1, 2, 5, 10, 15, 20, 25]
        expectations_tev = [1.9e5, 3.6e8, 5.0e8, 7.0e7, 4.1e6, 1.7e5, 5.7e3]
        expectations_pev = [1.5e1, 4.0e7, 5.1e8, 3.8e8, 6.0e7, 4.9e6, 2.8e5]
        expectations_eev = [3.7e-4, 1.4e6, 1.6e8, 6.5e8, 2.7e8, 4.4e7, 4.4e6]
        for i, length in enumerate(lengths):
            assert (arz_pulse.had_shower_profile(z=length, energy=1e3) * 1e6 / 1.602e-19
                    == pytest.approx(expectations_tev[i], rel=0.05))
            assert (arz_pulse.had_shower_profile(z=length, energy=1e6) * 1e3 / 1.602e-19
                    == pytest.approx(expectations_pev[i], rel=0.05))
            assert (arz_pulse.had_shower_profile(z=length, energy=1e9) / 1.602e-19
                    == pytest.approx(expectations_eev[i], rel=0.05))
        lengths = np.linspace(1, 15, 1000)
        for energy in [1e3, 1e6, 1e9]:
            print(np.max(arz_pulse.had_shower_profile(lengths, energy)) / 1.602e-19)
            assert (np.max(arz_pulse.had_shower_profile(lengths, energy)) / 1.602e-19
                    == pytest.approx(energy*0.6, rel=0.1))

    def test_max_length(self, arz_pulse):
        """Test max_length method"""
        energies = np.logspace(3, 9, 7)
        expectations = [5.3, 6.7, 8.0, 9.3, 10.6, 11.9, 13.2]
        for energy, expected in zip(energies, expectations):
            assert arz_pulse.max_length(energy) == pytest.approx(expected, rel=0.01)



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
