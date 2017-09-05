"""Tests of timing performance for different pieces of the pyrex package"""

import timeit
import numpy as np
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
                              center_frequency=250, bandwidth=100, resistance=None,
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

    t_loop += performance_test("pulse = AskaryanSignal(times=times, energy=p.energy*1e-3, "+
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

    t_loop += performance_test("pf.propagate(pulse)", repeats=10,
                               setup="import numpy as np;"+
                                     "import pyrex;"+
                                     "times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False);"+
                                     "pulse = pyrex.AskaryanSignal(times=times, "+
                                     "energy=p.energy*1e-3, theta=psi, n=n)",
                               use_globals={"p": p, "pf": pf, "psi": psi,
                                            "n": ice.index(p.vertex[2])})
    print("  * ~1000 antennas")

    t_loop += performance_test("ant.receive(pulse)", repeats=100,
                               setup="import numpy as np;"+
                                     "import pyrex;"+
                                     "times = np.linspace(-20e-9, 80e-9, 2048, endpoint=False);"+
                                     "pulse = pyrex.AskaryanSignal(times=times, "+
                                     "energy=p.energy*1e-3, theta=psi, n=n)",
                               use_globals={"p": p, "pf": pf, "psi": psi,
                                            "n": ice.index(p.vertex[2]),
                                            "ant": ant})
    print("  * ~1000 antennas")


    print("Total time: ", round(t+t_loop*800,1), "seconds per event on average")

if __name__ == '__main__':
    test_EventKernel_event(1e6)
    