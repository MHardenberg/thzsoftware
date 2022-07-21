import thzsoftware.fitting as fit
import thzsoftware.helper as h
import numpy as np


def test_linear_fit():
    tolerance = 1e-3
    a = 0.331
    b = -1.5

    xs = np.arange(100)
    ys = a*xs+b

    a_fit, b_fit = fit.linear_fit(xs, ys)

    assert h.within_tolerance(a_fit, a, tolerance) and h.within_tolerance(b, b_fit, tolerance)\
        , ValueError('Fit parameters not accurate.')
