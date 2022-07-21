import numpy as np


def within_tolerance(x, y, tolerance):
    allowed_types = (int, float, np.integer, np.inexact, np.ndarray)
    error1 = 'Both values must be of type int, float, numpy.integer, numpy.inexact or numpy.ndarray of length 1. ' \
             f'type(x) = {type(x)}. '
    error2 = 'Both values must be of type int, float, numpy.integer, numpy.inexact or numpy.ndarray of length 1. ' \
             f'type(y) = {type(y)}. '
    assert isinstance(x, allowed_types), TypeError(error1)
    assert isinstance(y, allowed_types), TypeError(error2)

    if isinstance(x, np.ndarray):
        assert len(x) == 1, TypeError(error1)
        x = float(x)
    if isinstance(y, np.ndarray):
        assert len(y) == 1, TypeError(error2)
        y = float(x)

    return np.abs(x - y) < tolerance


def create_mock_thz_pulse(n=100):
    # creates mock pulse to test other functions.
    assert n >= 50, ValueError("To create a reasonable pulse, use a longer base. E.g. n = 100.")
    ys = np.zeros(n)
    peak, peak_pos = 20, n // 2
    left_low, left_low_pos = -5, n // 2 - n // 20
    right_low, right_low_pos = -10, n // 2 + n // 10

    dist_left = peak_pos - left_low_pos
    dist_right = right_low_pos - peak_pos

    # construct left low
    ys[left_low_pos - dist_left:left_low_pos] = [x / dist_left * left_low for x in range(dist_left)]
    ys[left_low_pos:left_low_pos + dist_left // 2] = [left_low - 2 * x / dist_left * left_low
                                                      for x in range(dist_left // 2)]

    # construct dominant peak
    ys[peak_pos - dist_left // 2:peak_pos] = [2 * x / dist_left * peak for x in range(dist_left // 2)]
    ys[peak_pos:peak_pos + dist_right // 2] = [peak - 2 * x / dist_right * peak for x in range(dist_right // 2)]

    # construct right low
    ys[right_low_pos - dist_right // 2:right_low_pos] = [2 * x / dist_right * right_low
                                                         for x in range(dist_right // 2)]
    ys[right_low_pos:right_low_pos + dist_right // 2] = [right_low - 2 * x / dist_right * right_low
                                                         for x in range(dist_right // 2)]

    return ys, (left_low_pos, peak_pos, right_low_pos)
