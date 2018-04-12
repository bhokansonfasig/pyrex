"""Module containing reconstruction methods for IREX"""

from itertools import combinations
import numpy as np
import scipy.optimize
from pyrex.ray_tracing import RayTracer


def quick_vertex_reconstruction(detector, threshold=None,
                                get_waveform=lambda ant: ant.all_waveforms[0]):
    triggered_antennas = [ant for ant in detector
                          if ant.trigger(get_waveform(ant))]
    reco_antennas = [ant for ant in triggered_antennas
                     if np.max(get_waveform(ant).values)>threshold]
    reco_positions = [np.array(ant.position) for ant in reco_antennas]
    reco_times = get_xcorr_times([get_waveform(ant) for ant in reco_antennas])
    return bancroft_scan_vertex(reco_positions, reco_times)


def full_vertex_reconstruction(detector, threshold=None,
                               get_waveform=lambda ant: ant.all_waveforms[0]):
    triggered_antennas = [ant for ant in detector
                          if ant.trigger(get_waveform(ant))]
    reco_antennas = [ant for ant in triggered_antennas
                     if np.max(get_waveform(ant).values)>threshold]
    reco_positions = [np.array(ant.position) for ant in reco_antennas]
    reco_times = get_xcorr_times([get_waveform(ant) for ant in reco_antennas])
    return minimizer_vertex_reconstruction(reco_positions, reco_times)


def get_xcorr_times(waveforms):
    delays = [0]
    time_delay = 0
    for a, b in zip(waveforms[:-1], waveforms[1:]):
        c = scipy.signal.correlate(a.values/np.std(a.values),
                                   b.values/np.std(b.values))
        bin_delay = np.argmax(c) - max(len(a.times), len(b.times)) + 1
        time_delay -= bin_delay * a.dt + a.times[0] - b.times[0]
        delays.append(time_delay)
    return delays


def minimizer_vertex_reconstruction(positions, times, guess=None):
    if guess is None:
        guess = bancroft_scan_vertex(positions, times)[0]
    reco = scipy.optimize.minimize(least_squares, guess, args=(positions, times),
                                   method='Nelder-Mead')
    return reco.x, reco.fun, reco.success


def least_squares(vertex, positions, times, method='trace'):
    tofs = []
    for position in positions:
        rt = RayTracer(vertex, position)
        if vertex[2]<-3000:
            return np.nan
        if rt.exists:
            tofs.append(rt.solutions[0].tof)
        else:
            return np.nan
    residuals = [((time_2-time_1) - (tof_2-tof_1))*1e9
                 for (time_1, tof_1), (time_2, tof_2)
                 in combinations(zip(times, tofs), 2)]

    return np.sum(np.array(residuals)**2)


def bancroft_vertex(positions, times, velocity=3e8/1.755):
    # Method based on thesis of Thomas Meures and the following link:
    # http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_8_Linear_least_squares_orthogonal_matrices.pdf
    A = []
    b = []
    n_pairs = len(positions)-1
    if n_pairs<4:
        raise ValueError("Need at least 4 antenna pairs")
    for i in range(n_pairs):
        p_i = positions[i]
        p_j = positions[i+1]
        t_i = times[i]
        t_j = times[i+1]
        line = []
        for k in range(3):
            line.append(p_i[k] - p_j[k])
        line.append(-velocity**2 * (t_i - t_j))
        A.append(line)
        b.append((np.sum(p_i**2) - np.sum(p_j**2) - velocity**2*(t_i**2-t_j**2))/2)
    Q, R = scipy.linalg.qr(A, mode='economic')
    c = np.dot(Q.T, b)
    try:
        return np.dot(np.linalg.inv(R), c)
    except np.linalg.LinAlgError:
        raise ValueError("Points given all lie in the same plane")


def bancroft_scan_vertex(positions, times, velocity=3e8/1.755):
    # Method based on thesis of Thomas Meures and the following link:
    # http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_8_Linear_least_squares_orthogonal_matrices.pdf
    ts = np.linspace(-150*1e-9, -30150*1e-9, 301)
    n_pairs = len(positions)-1
    if n_pairs<3:
        raise ValueError("Need at least 3 antenna pairs")
    t0 = np.min(times)

    A = []
    b0 = []
    for i in range(n_pairs):
        p_i = positions[i]
        p_j = positions[i+1]
        t_i = times[i]-t0
        t_j = times[i+1]-t0
        line = []
        for k in range(3):
            line.append(p_i[k] - p_j[k])
        A.append(line)
        b0.append((np.sum(p_i**2) - np.sum(p_j**2)
                   - velocity**2*(t_i**2-t_j**2))/2)
    Q, R = scipy.linalg.qr(A, mode='economic')

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        raise ValueError("Points given all lie in the same plane")

    residuals = []
    vertices = []
    for t in ts:
        b = np.zeros(len(b0))
        for i in range(n_pairs):
            b[i] = b0[i] + velocity**2 * t*(times[i]-times[i+1])
        c = np.dot(Q.T, b)
        vertex = np.dot(R_inv, c)
        prod = np.dot(A, vertex)
        residuals.append((np.linalg.norm(b/np.linalg.norm(b) -
                                         prod/np.linalg.norm(prod))**2)/n_pairs)
        vertices.append(vertex)

    min_index = np.argmin(residuals)
    return vertices[min_index], ts[min_index], residuals[min_index]
