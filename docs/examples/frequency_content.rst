Askaryan Frequency Content
==========================

In this example we explore how the freqeuncy spectrum of an Askaryan pulse changes as a function of the off-cone angle (i.e. the angular distance between the Cherenkov angle and the observation angle). This code can be run from the Frequency Content notebook in the examples directory.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyrex

    # First, set the depth of the neutrino source and find the index of refraction
    # at that depth.
    # Then use that index of refraction to calculate the Cherenkov angle.
    depth = -1000
    n = pyrex.IceModel.index(depth)
    ch_angle = np.arcsin(np.sqrt(1 - 1/n**2))

    # Now, for a range of dthetas, generate an Askaryan pulse dtheta away from the
    # Chereknov angle and plot its frequency spectrum.
    for dtheta in np.radians(np.logspace(-1, 1, 5)):
        n_pts = 10001
        pulse = pyrex.AskaryanSignal(times=np.linspace(-20e-9, 80e-9, n_pts),
                                     energy=1e8, theta=ch_angle-dtheta, n=n)
        plt.plot(pulse.frequencies[:int(n_pts/2)] * 1e-6, # Convert from Hz to MHz
                 np.abs(pulse.spectrum)[:int(n_pts/2)])
        plt.title("Frequency Spectrum of Askaryan Pulse\n"+
                  str(round(np.degrees(dtheta),2))+" Degrees Off-Cone")
        plt.xlabel("Frequency (MHz)")
        plt.xlim(0, 3000)
        plt.show()

    # Actually, we probably really want to see the frequency content after the
    # signal has propagated through the ice a bit. So first set up the ray tracer
    # from our neutrino source to some other point where our antenna might be
    # (and make sure a path between those two points exists).
    rt = pyrex.RayTracer(from_point=(0, 0, depth), to_point=(500, 0, -100))
    if not rt.exists:
        raise ValueError("Path to antenna doesn't exist!")

    # Finally, plot the signal spectrum as it appears at the antenna position by
    # propagating it along the (first solution) path.
    path = rt.solutions[0]
    for dtheta in np.radians(np.logspace(-1, 1, 5)):
        n_pts = 2048
        pulse = pyrex.AskaryanSignal(times=np.linspace(-20e-9, 80e-9, n_pts),
                                     energy=1e8, theta=ch_angle-dtheta, n=n)
        path.propagate(pulse)
        plt.plot(pulse.frequencies[:int(n_pts/2)] * 1e-6, # Convert from Hz to MHz
                 np.abs(pulse.spectrum)[:int(n_pts/2)])
        plt.title("Frequency Spectrum of Askaryan Pulse\n"+
                  str(round(np.degrees(dtheta),2))+" Degrees Off-Cone")
        plt.xlabel("Frequency (MHz)")
        plt.xlim(0, 3000)
        plt.show()

    # You may notice the sharp cutoff in the frequency spectrum above 1 GHz.
    # This is due to the ice model, which defines the attenuation length in a
    # piecewise manner for frequencies above or below 1 GHz.