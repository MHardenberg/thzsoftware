import numpy as np
import scipy.fft as ft
import scipy.signal as signal
from scipy.constants import c
from thzsoftware import fitting as fit
from thzsoftware import math as mt


# fourier transform
def fft(amplitude_array, sample_rate, pad=None):
    assert isinstance(amplitude_array, (list, tuple, np.ndarray)), TypeError("'amplitude_array' must be array like. ("
                                                                             "list, tuple or numpy.array).")
    assert isinstance(sample_rate, (int, float)), TypeError("'sample_rate' must be int or float.")
    assert pad is None or isinstance(pad, int), TypeError("'pad' kwarg must be int number of right side "
                                                          "padding zeros or or None.")

    n = len(amplitude_array)
    if pad:
        amplitude_array = np.pad(amplitude_array, (0, pad), mode="constant")

    amplitude_transform = ft.fft(amplitude_array)[:n // 2]
    argument_transform = ft.fftfreq(n, sample_rate ** -1)[:n // 2]

    return argument_transform, amplitude_transform


# inverse Fourier transform
def ifft(amplitude_transform):
    assert isinstance(amplitude_transform, (list, tuple, np.ndarray)), TypeError("'amplitude_transform' must be array "
                                                                                 "like. ( "
                                                                                 "list, tuple or numpy.array).")
    amplitude = ft.ifft(amplitude_transform)
    return amplitude


# Phase unraveling
def compute_phase(zs, unwrap=True):  # takes in a complex array and returns modulus and unwrapped argument
    assert isinstance(unwrap, bool), TypeError(f"unwrap kwarg must be boolean. type(unwrap) = {type(unwrap)}.")
    r, phi = mt.carthesian_to_polar(zs)
    if unwrap:
        phi = mt.unwrap_angles(phi)

    return r, phi


# Phase extrapolation
def extrapolate_phase(fs, phase, f_limits):
    # This function cuts the data to satisfy the limits.
    # Then it fits the data, and forces the intersection to be 0.
    assert isinstance(fs, (list, tuple, np.ndarray)), TypeError(f"Input must be array-like. type(fs) = {type(fs)}")
    assert isinstance(phase, (list, tuple, np.ndarray)), TypeError("Input must be array-like."
                                                                   f" type(phase) = {type(phase)}")
    assert isinstance(f_limits, (list, tuple, np.ndarray)) and len(f_limits) == 2, \
        TypeError(f"Input must be array-like of length 2. type(f_limits) = {type(f_limits)},"
                  f" len(f_limits) = {len(f_limits)}")
    assert fs[-1] > f_limits[0] >= fs[0] and fs[-1] > f_limits[1] >= fs[0], ValueError("f_limits out of bounds."
                                                                                       f"f_limits = {f_limits}.")
    assert f_limits[0] < f_limits[1], ValueError("f_limits[0] must be lessen than f_limits[1]"
                                                 f"f_limits = {f_limits}.")

    fs_limited = fs[(fs > f_limits[0]) & (fs < f_limits[1])]
    phase_limited = phase[(fs > f_limits[0]) & (fs < f_limits[1])]

    fs_capped = fs[fs < f_limits[1]]
    extrapolation_length = len(fs_capped) - len(fs_limited)

    a, b = fit.linear_fit(fs_limited, phase_limited)

    # finally shift fs to the left.
    out = np.append(a * fs_capped[:extrapolation_length], phase[extrapolation_length:] - b)
    assert out.shape == fs.shape, f"Input shape not conserved. out.shape = {out.shape}, fs.shape = {fs.shape}."
    return out


# apply Tukey window function
def window_tukey(ts, ys, center_index, width, alpha=.2):
    dt = ts[1] - ts[0]
    points = int(max(1, width // dt))

    assert points < len(ts), f"Too few data points {len(dt)} to represent pulse of length {width}."
    assert points / 2 < center_index < len(ts) - points / 2, f"Window out of bound, left limit = " \
                                                             f"{center_index - (points // 2)} "

    left_win_lim = int(center_index - points // 2 - points // 4)  # to shift window just behind max
    window = np.zeros(ts.shape)
    window[left_win_lim:left_win_lim + points] = signal.tukey(points, alpha=alpha)
    return ys * window


def compute_n_by_phase(frequency, phase_air, phase_sample, distance, n0=1, tolerance=1e-16):
    for name, candidate in (("frequency", frequency), ("phase_air", phase_air), ("phase_sample", phase_sample)):
        assert isinstance(candidate, (list, tuple, np.ndarray)), TypeError("Input must be array-like."
                                                                           f" type({name}) ="
                                                                           f" {type(candidate)}")
    assert len(frequency) == len(phase_air) == len(phase_sample), ValueError("Input array must be same length.")
    assert isinstance(distance, (int, float, np.integer, np.floating)), TypeError("distance and n0 inputs must be int"
                                                                                  " or float. type(distance) ="
                                                                                  f" {type(distance)}, type(n0) ="
                                                                                  f" type{type(n0)}.")
    assert isinstance(tolerance, (int, float)), TypeError("'tolerance' kwarg must be int of float. type(tolerance) = "
                                                          f"{type(tolerance)}")

    fs = frequency.copy()
    poles = np.where(fs < tolerance)  # in case of f = 0, we return np.inf and avoid poles.
    fs[poles] = 1.  # this will be replaced later.

    # source: J. Neu and C. A. Schmuttenmaer "Tutorial: An introduction to teraherty time domain spectroscopy (tds)"
    out = np.abs(phase_air - phase_sample) * c / (2 * np.pi * fs * distance) + n0
    out[poles] = np.inf
    return out
