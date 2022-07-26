import numpy as np
from thzsoftware.tds import phase as pt
from thzsoftware import math as mt
from thzsoftware import helper as h


def test_fft():
    f_test = 7.3141
    xs = np.linspace(0, 10 * np.pi, 500)
    ys = np.sin(2 * np.pi * f_test * xs)
    sample_rate = xs[1] - xs[0]

    fs, yf = pt.fft(ys, sample_rate ** -1, pad=None)
    tolerance = fs[1] - fs[0]
    assert h.within_tolerance(fs[np.abs(yf) == np.max(np.abs(yf))], f_test, tolerance), ValueError(
        'Frequency component != '
        f'f_test = {f_test}')


def test_extrapolate_phase():
    tolerance = 1e-3
    a = 0.331
    b = -1.5

    limits = (25, 50)
    xs = np.arange(100)
    ys_original = a * xs + b
    ys_cut = np.append(np.zeros(limits[0]), ys_original[limits[0]:])

    ys_extrapolated = pt.extrapolate_phase(xs, ys_cut, limits)

    for i in range(len(xs)):
        assert h.within_tolerance(ys_original[i], ys_extrapolated[i] + b, tolerance), ValueError('Extrapolation '
                                                                                                 'inaccurate.')


def test_compute_phase():
    norm = np.array([0.87, 0.44, 0.43, 1, 2]).astype(complex)
    phase = np.array([0.23, 2, 3, 4.22, 5]).astype(complex)  # unwrapping will reverse

    zs = norm * np.exp(1j * phase)
    norm_out, phase_out = pt.compute_phase(zs)

    phase_last = phase_out[0]
    for x in range(1, len(norm)):
        assert phase_last > phase_out[x], ValueError("Phase not decreasing.")
        phase_last = phase_out[x]

        assert h.within_tolerance(norm_out[x], norm[x], 1e-4), ValueError("Modulus not reconstructed.")


def test_compute_n_by_phase():
    fs = np.arange(1, 11)
    phase_air = fs * 5.0
    phase_sample = fs * 1.5
    d = 10

    expected_out = 16699708.3873293
    out = pt.compute_n_by_phase(fs, phase_air, phase_sample, d, n0=.33)

    for i in range(10):
        assert h.within_tolerance(out[i],expected_out,1e-5), ValueError("n inaccurate.")
