import numpy as np
import scipy.fft as ft
import scipy.signal as signal
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
    argument_transform = ft.fftfreq(n, sample_rate**-1)[:n // 2]

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
