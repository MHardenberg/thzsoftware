import numpy as np
from thzsoftware import helper as h


def test_within_tolerance():
    x = 1
    y = 1.5

    tolerance_1 = 0.6
    tolerance_2 = 0.5

    assert h.within_tolerance(x, y, tolerance_1) and not h.within_tolerance(x, y, tolerance_2)


def test_create_mock_thz_pulse():
    ys, positions = h.create_mock_thz_pulse()
    assert isinstance(ys, np.ndarray), TypeError(f"Pulse should be numpy.ndarray type. type(ys) = {type(ys)}")
    assert isinstance(positions, tuple), TypeError("Secondary output should be tuple of indices for left_low,"
                                                   f"right_low and peak. type(positions) = {type(positions)}")
    for pos in positions:
        assert isinstance(pos, (int, np.integer)), TypeError("Position indices should be integers. "
                                                             f"type pos = {type(pos)}")

    left_low, peak, right_low = positions

    assert left_low == np.argmin(ys[:peak]), ValueError(f"left_low displaced. left_low = {left_low} and should be"
                                                        f"np.argmin(ys[:peak]) = {np.argmin(ys[:peak])}")
    assert peak == np.argmax(ys), ValueError(f"peak displaced. peak = {peak} and should be "
                                             f"np.argmax(ys) = {np.argmax(ys)}")
    assert right_low == peak + np.argmin(ys[peak:]), ValueError(f"right_low displaced. right_low = {right_low} and "
                                                                "should be "
                                                                f"np.argmin(ys[peak:]) = {peak + np.argmin(ys[peak:])}")
