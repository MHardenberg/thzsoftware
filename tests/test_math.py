import numpy as np
from thzsoftware import math as mt
from thzsoftware import helper as h


def test_shift_array():
    array = np.array([1, 2, 3, 4, 5])
    array_shifted_test = np.array([-1, -1, 1, 2, 3])
    array_shifted = mt.shift_array(array, 2, fill_value=-1)

    for i in range(len(array)):
        assert array_shifted[i] == array_shifted_test[i], ValueError("Array shift inaccurate.")


def test_cartesian_to_polar():
    tolerance = 1e-4
    rs_test = np.array([1, 2, 3, 4, 5])
    phis_test = np.array([.5, .4, .3, .2, .1])

    cartesian = np.array([rs_test[i] * (np.cos(phis_test[i]) + 1j * np.sin(phis_test[i])) for i in range(len(rs_test))])
    rs, phis = mt.carthesian_to_polar(cartesian)

    for i in range(len(rs)):
        assert h.within_tolerance(rs[i], rs_test[i], tolerance) and\
               h.within_tolerance(phis[i], phis_test[i], tolerance), ValueError('Polar forms not accurate.')


def test_unwrap_angles():
    angles_monotonic = -0.5 * np.arange(50)
    angles_wrapped = np.append(angles_monotonic[:25], angles_monotonic[25:] + 4 * np.pi)

    angles = mt.unwrap_angles(angles_wrapped)

    x0 = angles[0]
    for x in angles[1:]:
        assert x0 > x, ValueError("Angles not monotonically decreasing")
        x0 = x


def test_zero_crossing():
    a = [1, 2, 1, 1, -3, -4, 7, 8, 9, 10]
    # test "before"
    crossings = mt.zero_crossings(a)
    assert crossings[0] == 3 and crossings[1] == 5, ValueError(f"Crossing inaccurate. Should be [3,5] is {crossings}")

    # test "after"
    crossings = mt.zero_crossings(a, find_index="after")
    assert crossings[0] == 4 and crossings[1] == 6, ValueError(f"Crossing inaccurate. Should be [4,6] is {crossings}")
