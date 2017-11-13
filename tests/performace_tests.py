"""Tests of timing performance for different pieces of the pyrex package"""

import timeit
import numpy as np
from scipy.special import lambertw
import collections
import pyrex

def performance_test(method_string, number=1, setup="pass", use_globals=False,
                     repeats=1, alternate_title=None):
    if alternate_title is not None:
        print(alternate_title)
    else:
        print(method_string)

    if use_globals:
        if isinstance(use_globals, dict):
            timer = timeit.Timer(stmt=method_string, setup=setup, globals=use_globals)
        else:
            timer = timeit.Timer(stmt=method_string, setup=setup, globals=globals())
    else:
        timer = timeit.Timer(stmt=method_string, setup=setup, globals=None)

    t = min(timer.repeat(repeats, number))
    t /= number

    print("  ",end="")
    if t<1e-7:
        print(t*1e6, "microseconds", end=" ")
    elif t<1e-4:
        print(round(t*1e6,3), "microseconds", end=" ")
    elif t<0.1:
        print(round(t*1e3,3), "milliseconds", end=" ")
    else:
        print(round(t, 3), "seconds", end=" ")
    print("<",number*repeats,">", sep="")

    return t




def test_EventKernel_event(energy=1e6):
    t = 0

    t += performance_test("p = gen.create_particle()", number=100,
                          setup="import pyrex;"+
                                "gen = pyrex.ShadowGenerator(dx=10000, dy=10000, "+
                                "dz=2800, energy_generator=lambda: "+str(energy)+")")

    t += performance_test("n = ice.index(p.vertex[2])", number=1000,
                          setup="import pyrex;"+
                                "import numpy as np;"+
                                "z = np.random.random() * -2800;"+
                                "p = pyrex.Particle(vertex=(0,0,z), direction=(0,0,1), "+
                                "energy="+str(energy)+");"+
                                "ice = pyrex.IceModel();")

    t += performance_test("pf = PathFinder(ice, p.vertex, ant.position)", number=1000,
                          setup="import pyrex;"+
                                "from pyrex import PathFinder;"
                                "import numpy as np;"+
                                "z = np.random.random() * -2800;"+
                                "p = pyrex.Particle(vertex=(0,0,z), direction=(0,0,1), "+
                                "energy="+str(energy)+");"+
                                "ice = pyrex.IceModel();"+
                                "ant = pyrex.DipoleAntenna(name='ant', position=(5000,5000,-200), "+
                                "center_frequency=250, bandwidth=100, resistance=None, "+
                                "effective_height=1.0, trigger_threshold=3*(12e-6), noisy=False)")

    ice = pyrex.IceModel()
    ant = pyrex.DipoleAntenna(name='ant', position=(5000,5000,-200),
                              center_frequency=250e6, bandwidth=100e6, resistance=None,
                              effective_height=1, trigger_threshold=36e-6, noisy=False)
    def get_good_path():
        z = np.random.random() * -2800
        p = pyrex.Particle(vertex=(0,0,z), direction=(0,0,1), energy=energy)
        pf = pyrex.PathFinder(ice, p.vertex, ant.position)
        k = pf.emitted_ray
        epol = np.vdot(k, p.direction) * k - p.direction
        epol = epol / np.linalg.norm(epol)
        psi = np.arccos(np.vdot(p.direction, k))
        if pf.exists and psi<np.pi/2:
            return p, pf, psi
        else:
            return get_good_path()

    p, pf, psi = get_good_path()

    t_loop = 0

    t_loop += performance_test("pf.exists", number=1000, use_globals={"pf": pf})
    print("  * ~1000 antennas")

    t_loop += performance_test("k = pf.emitted_ray; "+
                               "epol = np.vdot(k, p.direction) * k - p.direction; "+
                               "epol = epol / np.linalg.norm(epol); "+
                               "psi = np.arccos(np.vdot(p.direction, k))", number=1000,
                               setup="import numpy as np",
                               use_globals={"p": p, "pf": pf},
                               alternate_title="Set k, epol, psi")
    print("  * ~1000 antennas")

    t_loop += performance_test("times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)",
                               number=1000, setup="import numpy as np")
    print("  * ~1000 antennas")

    t_loop += performance_test("pulse = AskaryanSignal(times=times, energy=p.energy, "+
                               "theta=psi, n=n)", number=100,
                               setup="import numpy as np;"+
                                     "from pyrex import AskaryanSignal;"+
                                     "times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)",
                               use_globals={"p": p, "pf": pf, "psi": psi,
                                            "n": ice.index(p.vertex[2])})
    print("  * ~1000 antennas")

    times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)
    pulse = pyrex.AskaryanSignal(times=times, energy=p.energy*1e-3,
                                 theta=psi, n=ice.index(p.vertex[2]))

    t_loop += performance_test("pf.propagate(pulse)", repeats=100,
                               setup="import numpy as np;"+
                                     "import pyrex;"+
                                     "times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False);"+
                                     "pulse = pyrex.AskaryanSignal(times=times, "+
                                     "energy=p.energy, theta=psi, n=n)",
                               use_globals={"p": p, "pf": pf, "psi": psi,
                                            "n": ice.index(p.vertex[2])})
    print("  * ~1000 antennas")

    t_loop += performance_test("ant.receive(pulse)", repeats=100,
                               setup="import numpy as np;"+
                                     "import pyrex;"+
                                     "times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False);"+
                                     "pulse = pyrex.AskaryanSignal(times=times, "+
                                     "energy=p.energy, theta=psi, n=n)",
                               use_globals={"p": p, "pf": pf, "psi": psi,
                                            "n": ice.index(p.vertex[2]),
                                            "ant": ant})
    print("  * ~1000 antennas")


    print("Total time:", round(t+t_loop*800,1), "seconds per event on average")



def test_PathFinder_propagate():
    t = 0

    pf = pyrex.PathFinder(pyrex.IceModel(), (0,0,-2800), (5000,5000,-200))
    while not(pf.exists):
        z = np.random.random()*-2800
        pf = pyrex.PathFinder(pyrex.IceModel(), (0,0,z), (5000,5000,-200))

    p = pyrex.Particle(vertex=pf.from_point,
                       direction=(1/np.sqrt(2),1/np.sqrt(2),0), energy=1e6)

    k = pf.emitted_ray
    epol = np.vdot(k, p.direction) * k - p.direction
    epol = epol / np.linalg.norm(epol)
    psi = np.arccos(np.vdot(p.direction, k))
    n = pyrex.IceModel.index(p.vertex[2])

    pulse = pyrex.AskaryanSignal(times=np.linspace(-20e-9, 80e-9, 2048, endpoint=False),
                                 energy=p.energy, theta=psi, n=n)

    t += performance_test("signal.values *= 1 / pf.path_length", repeats=100,
                          setup="import pyrex;"+
                                "signal = pyrex.Signal(pulse.times, pulse.values)",
                          use_globals={"pf": pf, "pulse": pulse})

    pulse.values *= 1 / pf.path_length

    t += performance_test("signal.filter_frequencies(pf.attenuation)", repeats=100,
                          setup="import pyrex;"+
                                "signal = pyrex.Signal(pulse.times, pulse.values)",
                          use_globals={"pf": pf, "pulse": pulse})

    # pulse.filter_frequencies(pf.attenuation)

    t += performance_test("signal.times += pf.tof", repeats=100,
                          setup="import pyrex;"+
                                "signal = pyrex.Signal(pulse.times, pulse.values)",
                          use_globals={"pf": pf, "pulse": pulse})

    print("Total time:", round(t*1000, 1), "milliseconds per signal")



def test_filter_attenuation():
    # filtered_spectrum = self.spectrum
    # responses = np.array(freq_response(self.frequencies))
    # filtered_spectrum *= responses
    # self.values = np.real(scipy.fftpack.ifft(filtered_spectrum))

    # freq_response = pf.attenuation
    # self = AskaryanSignal(times=np.linspace(-20e-9, 80e-9, 2048, endpoint=False),
    #                       energy=p.energy*1e-3, theta=psi, n=n)
    t = 0

    pf = pyrex.PathFinder(pyrex.IceModel(), (0,0,-2800), (5000,5000,-200))
    while not(pf.exists):
        z = np.random.random()*-2800
        pf = pyrex.PathFinder(pyrex.IceModel(), (0,0,z), (5000,5000,-200))

    p = pyrex.Particle(vertex=pf.from_point,
                       direction=(1/np.sqrt(2),1/np.sqrt(2),0), energy=1e6)

    k = pf.emitted_ray
    epol = np.vdot(k, p.direction) * k - p.direction
    epol = epol / np.linalg.norm(epol)
    psi = np.arccos(np.vdot(p.direction, k))
    n = pyrex.IceModel.index(p.vertex[2])

    pulse = pyrex.AskaryanSignal(times=np.linspace(-20e-9, 80e-9, 2048, endpoint=False),
                                 energy=p.energy, theta=psi, n=n)

    t += performance_test("filtered_spectrum = pulse.spectrum", number=1000,
                          use_globals={"pulse": pulse})

    t += performance_test("fs = pulse.frequencies", number=1000,
                          use_globals={"pulse": pulse})

    fs = pulse.frequencies

    # performance_test("alen = ice.attenuation_length(-1000, fa*1e-6)", number=100,
    #                  use_globals={"ice": pyrex.IceModel(), "fa": np.abs(fs)})

    t += performance_test("responses = freq_response(fs)", number=1000,
                          use_globals={"freq_response": pf.attenuation, "fs": fs})

    # t += performance_test("responses = np.array(freq_response(pulse.frequencies))",
    #                       number = 100, setup="import numpy as np",
    #                       use_globals={"freq_response": pf.attenuation,
    #                                    "pulse": pulse})

    filtered_spectrum = pulse.spectrum * np.array(pf.attenuation(pulse.frequencies))

    t += performance_test("np.real(scipy.fftpack.ifft(filtered_spectrum))",
                          number=1000, setup="import numpy as np; import scipy.fftpack",
                          use_globals={"filtered_spectrum": filtered_spectrum})

    print("Total time:", round(t*1000, 1), "milliseconds for", len(fs), "frequencies")
    print("           ", round(t*1e6/len(fs), 1), "microseconds per frequency")



def test_tof_methods():
    from_pt = np.array((-5000,-5000,0))
    to_pt = np.array((5000,5000,-2800))
    def step_method(from_point, to_point, ice, n_steps):
        t = 0
        z0 = from_point[2]
        z1 = to_point[2]
        u = to_point - from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dz = z1 - z0
        drdz = rho / dz
        dz /= n_steps
        for i in range(n_steps):
            z = z0 + (i+0.5)*dz
            dr = drdz * dz
            p = np.sqrt(dr**2 + dz**2)
            t += p / 3e8 * ice.index(z)
        return t

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=10)", number=1000,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": step_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, pyrex.IceModel(), n_steps=10))

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=1000)", number=100,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": step_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, pyrex.IceModel(), n_steps=1000))

    def trapz_method(from_point, to_point, ice, n_steps):
        z0 = from_point[2]
        z1 = to_point[2]
        zs = np.linspace(z0, z1, n_steps, endpoint=True)
        u = to_point - from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        integrand = ice.index(zs)
        t = np.trapz(integrand, zs) / 3e8 * np.sqrt(1 + (rho / (z1 - z0))**2)
        return np.abs(t)

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=10)", number=1000,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": trapz_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", trapz_method(from_pt, to_pt, pyrex.IceModel(), n_steps=10))

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=1000)", number=100,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": trapz_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", trapz_method(from_pt, to_pt, pyrex.IceModel(), n_steps=1000))

    def sum_method(from_point, to_point, ice, n_steps):
        z0 = from_point[2]
        z1 = to_point[2]
        dz = (z1 - z0) / n_steps
        zs, dz = np.linspace(z0+dz/2, z1+dz/2, n_steps, endpoint=False, retstep=True)
        u = to_point - from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dr = rho / (z1 - z0) * dz
        dp = np.sqrt(dz**2 + dr**2)
        ts = dp / 3e8 * ice.index(zs)
        t = np.sum(ts)
        return t

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=10)", number=1000,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": sum_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", sum_method(from_pt, to_pt, pyrex.IceModel(), n_steps=10))

    performance_test("tof(from_pt, to_pt, IceModel(), n_steps=1000)", number=100,
                     setup="from pyrex import IceModel",
                     use_globals={"tof": sum_method,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", sum_method(from_pt, to_pt, pyrex.IceModel(), n_steps=1000))


def test_atten_methods():
    from_pt = np.array((-5000,-5000,0))
    to_pt = np.array((5000,5000,-2800))
    fs = np.logspace(1, 5, num=5)
    freqs = np.append(np.append([0], fs), -fs)
    def step_method(from_point, to_point, freqs, ice, n_steps):
        fa = np.abs(freqs)
        atten = 1
        z0 = from_point[2]
        z1 = to_point[2]
        u = to_point - from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dz = z1 - z0
        drdz = rho / dz
        dz /= n_steps
        for i in range(n_steps):
            z = z0 + (i+0.5)*dz
            dr = drdz * dz
            p = np.sqrt(dr**2 + dz**2)
            alen = ice.attenuation_length(z, fa)
            atten *= np.exp(-p/alen)
        return atten

    performance_test("atten(from_pt, to_pt, freqs, IceModel(), n_steps=10)",
                     number=1000, setup="from pyrex import IceModel",
                     use_globals={"atten": step_method, "freqs": freqs,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, freqs, pyrex.IceModel(), n_steps=10))

    performance_test("atten(from_pt, to_pt, freqs, IceModel(), n_steps=1000)",
                     number=10, setup="from pyrex import IceModel",
                     use_globals={"atten": step_method, "freqs": freqs,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, freqs, pyrex.IceModel(), n_steps=1000))

    def prod_method(from_point, to_point, freqs, ice, n_steps):
        fa = np.abs(freqs)
        z0 = from_point[2]
        z1 = to_point[2]
        zs, dz = np.linspace(z0, z1, n_steps, endpoint=True, retstep=True)
        u = to_point - from_point
        rho = np.sqrt(u[0]**2 + u[1]**2)
        dr = rho / (z1 - z0) * dz
        dp = np.sqrt(dz**2 + dr**2)
        alens = ice.attenuation_length(zs, fa)
        attens = np.exp(-dp/alens)
        return np.prod(attens, axis=0)

    performance_test("atten(from_pt, to_pt, freqs, IceModel(), n_steps=10)",
                     number=1000, setup="from pyrex import IceModel",
                     use_globals={"atten": prod_method, "freqs": freqs,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, freqs, pyrex.IceModel(), n_steps=10))

    performance_test("atten(from_pt, to_pt, freqs, IceModel(), n_steps=1000)",
                     number=100, setup="from pyrex import IceModel",
                     use_globals={"atten": prod_method, "freqs": freqs,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, freqs, pyrex.IceModel(), n_steps=1000))

    performance_test("atten(from_pt, to_pt, freqs, IceModel(), n_steps=100000)",
                     number=1, setup="from pyrex import IceModel",
                     use_globals={"atten": prod_method, "freqs": freqs,
                                  "from_pt": from_pt, "to_pt": to_pt})
    print("Returns:", step_method(from_pt, to_pt, freqs, pyrex.IceModel(), n_steps=100000))


def test_alen_differentiation_methods():
    z = -500
    zs = np.linspace(-1000, 0, 101)
    f = 1e8
    fs = np.logspace(1, 11, 101)

    def try_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        try:
            a_lens = np.zeros(len(f))
        except TypeError:
            # f is a scalar, so return single value or array based on depths
            # (automatic so long as z is a numpy array)
            a, b = pyrex.IceModel.coeffs(t, f)
            return np.exp(-(a + b * w))

        try:
            a_lens = np.zeros((len(z),len(f)))
        except TypeError:
            # z is a scalar, so return array based on frequencies
            for i, freq in enumerate(f):
                a, b = pyrex.IceModel.coeffs(t, freq)
                a_lens[i] = np.exp(-(a + b * w[i]))
            return a_lens

        # f and z are both arrays, so return 2-D array of values
        for i, freq in enumerate(f):
            a, b = pyrex.IceModel.coeffs(t, freq)
            a_lens[:,i] = np.exp(-(a + b * w[i]))
        return a_lens

    print("'try' method:")
    t = performance_test("alen(z, f)", number=1000,
                         use_globals={"alen": try_method, "z": z, "f": f})
    print(" ", round(t*1e6, 3), "microseconds per pair")

    t = performance_test("alen(zs, f)", number=1000,
                         use_globals={"alen": try_method, "zs": zs, "f": f})
    print(" ", round(t*1e6/len(zs), 3), "microseconds per pair")

    t = performance_test("alen(z, fs)", number=1000,
                         use_globals={"alen": try_method, "z": z, "fs": fs})
    print(" ", round(t*1e6/len(fs), 3), "microseconds per pair")

    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": try_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")


    def hasattr_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        if hasattr(z, "__len__") and hasattr(f, "__len__"):
            # f and z are both arrays, so return 2-D array of values
            a_lens = np.zeros((len(z),len(f)))
            for i, freq in enumerate(f):
                a, b = pyrex.IceModel.coeffs(t, freq)
                a_lens[:,i] = np.exp(-(a + b * w[i]))
            return a_lens
        elif hasattr(f, "__len__"):
            # z is a scalar, so return array based on frequencies
            a_lens = np.zeros(len(f))
            for i, freq in enumerate(f):
                a, b = pyrex.IceModel.coeffs(t, freq)
                a_lens[i] = np.exp(-(a + b * w[i]))
            return a_lens
        else:
            # f is a scalar, so return single value or array based on depths
            # (automatic so long as z is a numpy array)
            a, b = pyrex.IceModel.coeffs(t, f)
            return np.exp(-(a + b * w))

    print("\n'hasattr' method:")
    t = performance_test("alen(z, f)", number=1000,
                         use_globals={"alen": hasattr_method, "z": z, "f": f})
    print(" ", round(t*1e6, 3), "microseconds per pair")

    t = performance_test("alen(zs, f)", number=1000,
                         use_globals={"alen": hasattr_method, "zs": zs, "f": f})
    print(" ", round(t*1e6/len(zs), 3), "microseconds per pair")

    t = performance_test("alen(z, fs)", number=1000,
                         use_globals={"alen": hasattr_method, "z": z, "fs": fs})
    print(" ", round(t*1e6/len(fs), 3), "microseconds per pair")

    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": hasattr_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")


    def isinstance_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        if isinstance(z, collections.Sequence):
            z = np.array(z)
        t = pyrex.IceModel.temperature(z)

        if isinstance(z, np.ndarray) and isinstance(f, (np.ndarray, collections.Sequence)):
            # f and z are both arrays, so return 2-D array of values
            a_lens = np.zeros((len(z),len(f)))
            for i, freq in enumerate(f):
                a, b = pyrex.IceModel.coeffs(t, freq)
                a_lens[:,i] = np.exp(-(a + b * w[i]))
            return a_lens
        elif isinstance(f, np.ndarray):
            # z is a scalar, so return array based on frequencies
            a_lens = np.zeros(len(f))
            for i, freq in enumerate(f):
                a, b = pyrex.IceModel.coeffs(t, freq)
                a_lens[i] = np.exp(-(a + b * w[i]))
            return a_lens
        else:
            # f is a scalar, so return single value or array based on depths
            # (automatic so long as z is a numpy array)
            a, b = pyrex.IceModel.coeffs(t, f)
            return np.exp(-(a + b * w))

    print("\n'isinstance' method:")
    t = performance_test("alen(z, f)", number=1000,
                         use_globals={"alen": isinstance_method, "z": z, "f": f})
    print(" ", round(t*1e6, 3), "microseconds per pair")

    t = performance_test("alen(zs, f)", number=1000,
                         use_globals={"alen": isinstance_method, "zs": zs, "f": f})
    print(" ", round(t*1e6/len(zs), 3), "microseconds per pair")

    t = performance_test("alen(z, fs)", number=1000,
                         use_globals={"alen": isinstance_method, "z": z, "fs": fs})
    print(" ", round(t*1e6/len(fs), 3), "microseconds per pair")

    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": isinstance_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")


    def force_method(z, f):
        z = np.array(z, ndmin=1)
        f = np.array(f, ndmin=1)
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        a_lens = np.zeros((len(z),len(f)))
        for i, freq in enumerate(f):
            a, b = pyrex.IceModel.coeffs(t, freq)
            a_lens[:,i] = np.exp(-(a + b * w[i]))
        return a_lens

    print("\nforce method:")
    t = performance_test("alen(z, f)", number=1000,
                         use_globals={"alen": force_method, "z": z, "f": f})
    print(" ", round(t*1e6, 3), "microseconds per pair")

    t = performance_test("alen(zs, f)", number=1000,
                         use_globals={"alen": force_method, "zs": zs, "f": f})
    print(" ", round(t*1e6/len(zs), 3), "microseconds per pair")

    t = performance_test("alen(z, fs)", number=1000,
                         use_globals={"alen": force_method, "z": z, "fs": fs})
    print(" ", round(t*1e6/len(fs), 3), "microseconds per pair")

    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": force_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")



def test_alen_calculation_methods():
    zs = np.linspace(-1000, 0, 101)
    fs = np.logspace(1, 11, 101)

    def np_slice_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        a_lens = np.zeros((len(z),len(f)))
        for i, freq in enumerate(f):
            a, b = pyrex.IceModel.coeffs(t, freq)
            a_lens[:,i] = np.exp(-(a + b * w[i]))
        return a_lens

    print("numpy slice method:")
    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": np_slice_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")


    def loop_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        a_lens = np.zeros((len(z),len(f)))
        for i, freq in enumerate(f):
            a, b = pyrex.IceModel.coeffs(t, freq)
            for j, depth in enumerate(z):
                a_lens[j,i] = np.exp(-(a[j] + b[j] * w[i]))
        return a_lens

    print("\nloop method:")
    t = performance_test("alen(zs, fs)", number=10,
                         use_globals={"alen": loop_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")


    def array_method(z, f):
        with np.errstate(divide='ignore'):
            w = np.log(f*1e-9)
        t = pyrex.IceModel.temperature(z)

        a, b = pyrex.IceModel.coeffs(t, f)
        a_lens = np.exp(-(a + b * w))
        return a_lens

    print("\narray method:")
    t = performance_test("alen(zs, fs)", number=1000,
                         use_globals={"alen": array_method, "zs": zs, "fs": fs})
    print(" ", round(t*1e6/len(zs)/len(fs), 3), "microseconds per pair")



def test_event_generation(energy):
    generator = pyrex.ShadowGenerator(10000, 10000, 2800, lambda: energy)
    print("energy =", energy)
    performance_test("gen.create_particle()", number=100,
                     use_globals={"gen": generator})


def test_angle_calculation():
    performance_test("r = np.sqrt(np.sum(np.array(origin)**2)); "
                     +"theta=np.arccos(origin[2]/r); "
                     +"phi=np.arctan(origin[1]/origin[0])",
                     setup="import numpy as np; origin=np.random.rand(3)*1000",
                     repeats=10000)

    performance_test("x, y, z = origin; r=np.sqrt(x**2+y**2+z**2); "
                     +"theta=np.arccos(z/r); "
                     +"phi=np.arctan(y/x)",
                     setup="import numpy as np; origin=np.random.rand(3)*1000",
                     repeats=10000)

    performance_test("x, y, z = origin; r=np.sqrt(np.sum(np.array(origin)**2)); "
                     +"theta=np.arccos(z/r); "
                     +"phi=np.arctan(y/x)",
                     setup="import numpy as np; origin=np.random.rand(3)*1000",
                     repeats=10000)

    performance_test("x, y, z = origin; "
                     +"theta=np.arccos(z/np.sqrt(x**2+y**2+z**2)); "
                     +"phi=np.arctan(y/x)",
                     setup="import numpy as np; origin=np.random.rand(3)*1000",
                     repeats=10000)


def test_normalization():
    def norm0(vector):
        return np.array(vector) / np.linalg.norm(vector)

    def norm1(vector):
        v = np.array(vector)
        if np.all(v==0):
            return v
        else:
            return v / np.linalg.norm(v)

    def norm2(vector):
        v = np.array(vector)
        mag = np.linalg.norm(v)
        if mag==0:
            return v
        else:
            return v / mag

    performance_test("norm0(vector)", repeats=10000,
                     setup="import numpy as np; vector=list(np.random.rand(3)*1000)",
                     use_globals={"norm0": norm0})

    performance_test("norm1(vector)", repeats=10000,
                     setup="import numpy as np; vector=list(np.random.rand(3)*1000)",
                     use_globals={"norm1": norm1})

    performance_test("norm2(vector)", repeats=10000,
                     setup="import numpy as np; vector=list(np.random.rand(3)*1000)",
                     use_globals={"norm2": norm2})


from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Unit import *

def test_pyspice_envelope():
    times, dt = np.linspace(0, 100e-9, 2048, endpoint=False, retstep=True)
    pulse = pyrex.AskaryanSignal(times=times, energy=1e8, theta=45*np.pi/180,
                                 n=1.75, t0=20e-9)

    performance_test("pyrex.Signal(pulse.times, pulse.envelope)", number=1000,
                     setup="import pyrex", use_globals={"pulse": pulse})

    spice_library = SpiceLibrary("/Users/bhokansonfasig/Documents/IceCube/"+
                                 "scalable_radio_array/spice_models")

    class NgSpiceSharedSignal(NgSpiceShared):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._signal = None

        def get_vsrc_data(self, voltage, time, node, ngspice_id):
            self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
            voltage[0] = np.interp(time, self._signal.times, self._signal.values)
            return 0

    ngspice_shared = NgSpiceSharedSignal()

    class NgSpiceSignal:
        def __init__(self, signal, shared=ngspice_shared):
            self.shared = ngspice_shared
            self.shared._signal = signal

    def setup_circuit(kind="biased"):
        if kind=="biased":
            circuit = Circuit('Biased Envelope Circuit')
            circuit.include(spice_library['hsms'])

            circuit.V('in', 'input', circuit.gnd, 'dc 0 external')
            # bias portion
            circuit.C(2, 'input', 1, 10@u_nF)
            circuit.R(2, 1, 2, 1@u_kOhm)
            circuit.X('D2', 'hsms', 2, circuit.gnd)
            circuit.R(3, 2, 'bias', 1@u_kOhm)
            circuit.V('bias', 'bias', circuit.gnd, 5@u_V)
            # envelope portion
            circuit.X('D1', 'hsms', 1, 'output')
            circuit.C(1, 'output', circuit.gnd, 220@u_pF)
            circuit.R(1, 'output', circuit.gnd, 50@u_Ohm)
            return circuit

        elif kind=="basic":
            circuit = Circuit('Biased Envelope Circuit')
            circuit.include(spice_library['hsms'])

            circuit.V('in', 'input', circuit.gnd, 'dc 0 external')
            # envelope portion
            circuit.X('D1', 'hsms', 'input', 'output')
            circuit.C(1, 'output', circuit.gnd, 220@u_pF)
            circuit.R(1, 'output', circuit.gnd, 50@u_Ohm)
            return circuit

        elif kind=="diode":
            circuit = Circuit('Diode Output')
            circuit.include(spice_library['hsms'])
            
            circuit.V('in', 'input', circuit.gnd, 'dc 0 external')
            circuit.X('D1', 'hsms', 'input', 'output')
            return circuit

    performance_test("ng_in = NgSpiceSignal(pulse); "+
                     "simulator = circuit.simulator(temperature=25, "+
                     "nominal_temperature=25, ngspice_shared=ng_in.shared); "+
                     "analysis = simulator.transient(step_time=dt, "+
                     "end_time=pulse.times[-1]); "+
                     "pyrex.Signal(analysis.output.abscissa, analysis.output)",
                     number=10,
                     setup="import pyrex; circuit = setup_circuit()",
                     use_globals={"pulse": pulse, "dt": dt,
                                  "setup_circuit": setup_circuit,
                                  "NgSpiceSignal": NgSpiceSignal},
                     alternate_title="pyrex.Signal(analysis.output.abscissa, "+
                                     "analysis.output)")


    def envelope_circuit(signal, cap=220e-12, res=50):
        v_c = 0
        v_out = []

        r_d = 25
        i_s = 3e-6
        n = 1.06
        v_t = 26e-3

        charge_exp = np.exp(-signal.dt/(res*cap))
        discharge = i_s*res*(1-charge_exp)
        lambert_factor = n*v_t*res/r_d*(1-charge_exp)
        frac = i_s*r_d/n/v_t

        lambert_exponent = np.log(frac) + frac

        for v_in in signal.values:
            a = lambert_exponent + (v_in - v_c)/n/v_t
            if a>100:
                b = np.log(a)
                lambert_term = a - b + b/a
            else:
                lambert_term = np.real(lambertw(np.exp(a)))
                if np.isnan(lambert_term):
                    lambert_term = 0
            v_c = v_c*charge_exp - discharge + lambert_factor*lambert_term
            v_out.append(v_c)

        return pyrex.Signal(signal.times, v_out,
                            value_type=pyrex.Signal.ValueTypes.voltage)


    performance_test("envelope(pulse)",
                     number=100,
                     use_globals={"pulse": pulse, "envelope": envelope_circuit},
                     alternate_title="analytic circuit simulation")


    # performance_test("ng_in = NgSpiceSignal(pulse); "+
    #                  "simulator = circuit.simulator(temperature=25, "+
    #                  "nominal_temperature=25, ngspice_shared=ng_in.shared); "+
    #                  "analysis = simulator.transient(step_time=dt, "+
    #                  "end_time=pulse.times[-1]); "+
    #                  "pyrex.Signal(analysis.output.abscissa, analysis.output)",
    #                  number=10,
    #                  setup="import pyrex; circuit = setup_circuit('diode')",
    #                  use_globals={"pulse": pulse, "dt": dt,
    #                               "setup_circuit": setup_circuit,
    #                               "NgSpiceSignal": NgSpiceSignal},
    #                  alternate_title="process diode output only")



def test_noise_generation():
    f_band = (100e6, 400e6)
    short_times = np.linspace(0, 100e-9, 101)
    long_times = np.linspace(0, 100e-9, 10001)
    performance_test("ThermalNoise(short_times, f_band, "
                     +"rms_voltage=1, n_freqs=100)", number=1000,
                     setup="from pyrex import ThermalNoise",
                     use_globals={"short_times": short_times,
                                  "f_band": f_band})

    performance_test("ThermalNoise(long_times, f_band, "
                     +"rms_voltage=1, n_freqs=100)", number=1000,
                     setup="from pyrex import ThermalNoise",
                     use_globals={"long_times": long_times,
                                  "f_band": f_band})

    performance_test("ThermalNoise(short_times, f_band, "
                     +"rms_voltage=1, n_freqs=10000)", number=100,
                     setup="from pyrex import ThermalNoise",
                     use_globals={"short_times": short_times,
                                  "f_band": f_band})

    performance_test("ThermalNoise(long_times, f_band, "
                     +"rms_voltage=1, n_freqs=10000)", number=10,
                     setup="from pyrex import ThermalNoise",
                     use_globals={"long_times": long_times,
                                  "f_band": f_band})


def test_antenna_noise_generation():
    def reset_antennas():
        times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False)
        pulse = pyrex.AskaryanSignal(times=times, energy=1e8, theta=np.radians(45))

        times2 = np.linspace(180e-9, 280e-9, 2048, endpoint=False)
        pulse2 = pyrex.AskaryanSignal(times=times, energy=1e8, theta=np.radians(45),
                                    t0=200e-9)

        antenna = pyrex.Antenna((0,0,0), freq_range=(100e6, 400e6), noise_rms=25e-6)
        antenna.receive(pulse)

        antenna2 = pyrex.Antenna((0,0,0), freq_range=(100e6, 400e6), noise_rms=25e-6)
        antenna2.receive(pulse)
        antenna2.receive(pulse2)

        return antenna, antenna2

    antenna, antenna2 = reset_antennas()

    performance_test("antenna.waveforms", number=1,
                     use_globals={"antenna": antenna})
    
    performance_test("antenna2.waveforms", number=1,
                     use_globals={"antenna2": antenna2})

    performance_test("antenna.waveforms", number=1000,
                     use_globals={"antenna": antenna})
    
    performance_test("antenna2.waveforms", number=1000,
                     use_globals={"antenna2": antenna2})


    antenna, antenna2 = reset_antennas()

    performance_test("antenna.is_hit", number=1,
                     use_globals={"antenna": antenna})

    performance_test("antenna2.is_hit", number=1,
                     use_globals={"antenna2": antenna2})

    performance_test("antenna.is_hit", number=1000,
                     use_globals={"antenna": antenna})

    performance_test("antenna2.is_hit", number=1000,
                     use_globals={"antenna2": antenna2})

    antenna, antenna2 = reset_antennas()

    performance_test("antenna.is_hit_during(np.linspace(0,300e-9,3001))",
                     number=1, setup="import numpy as np",
                     use_globals={"antenna": antenna})

    performance_test("antenna2.is_hit_during(np.linspace(0,300e-9,3001))",
                     number=1, setup="import numpy as np",
                     use_globals={"antenna2": antenna2})

    performance_test("antenna.is_hit_during(np.linspace(0,300e-9,3001))",
                     number=1, setup="import numpy as np",
                     use_globals={"antenna": antenna})

    performance_test("antenna2.is_hit_during(np.linspace(0,300e-9,3001))",
                     number=1, setup="import numpy as np",
                     use_globals={"antenna2": antenna2})


if __name__ == '__main__':
    # test_EventKernel_event(1e6)
    # print()
    # test_EventKernel_event(1e8)
    # print()
    # test_EventKernel_event(1e10)
    # print()

    # test_PathFinder_propagate()
    # test_filter_attenuation()

    # test_tof_methods()
    # test_atten_methods()
    # test_alen_differentiation_methods()
    # test_alen_calculation_methods()

    # test_angle_calculation()

    # test_normalization()

    # print("\nEnvelope:")
    # test_pyspice_envelope()

    # test_noise_generation()

    test_antenna_noise_generation()
