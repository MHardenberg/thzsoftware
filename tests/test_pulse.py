import numpy as np
from thzsoftware.tds import pulse as pt
from thzsoftware import math as mt
from thzsoftware import helper as h


def test_convolve():
    array1 = np.zeros(50)
    array1[10:15] = 1

    length = 50
    array2 = np.zeros(length)
    array2[20:25] = 1
    xs, conv = pt.convolve(array1, array2)

    assert conv[conv == np.max(conv)] == conv[length - 10], ValueError("Convolution displaced.")
    assert np.min(xs) == -length and np.max(xs) == length - 1


def test_define_thz_pulses():
    # test assumes noiseless signal (generated)
    n = 100
    pulse = h.create_mock_thz_pulse(n)[0]
    # find limits
    limits = pt.define_thz_pulses(pulse)

    pulse_rev = -pulse
    limits_rev = pt.define_thz_pulses(pulse_rev)

    assert limits[0] == limits_rev[0] and limits[1] == limits_rev[1], ValueError("Define pulse function should be"
                                                                                 " robust against global sign changes."
                                                                                 f" limits = {limits}, limits_rev"
                                                                                 f" = {limits_rev}")

    # square pulse to integrate intensity
    pulse = pulse**2
    rel_difference = np.sum(pulse) - np.sum(pulse[limits[0]:limits[1]+1])
    rel_difference /= np.sum(pulse)
    print(rel_difference)
    coverage = 1 + (np.sum(pulse[limits[0]:limits[1]]) - np.sum(pulse)) / np.sum(pulse)
    assert coverage <= 1, ValueError("Coverage cannot exceed 1.")
    assert h.within_tolerance(coverage, 1, 5*1e-2), ValueError("Not enough pulse coverage.")


def test_find_pulse():
    length = 50
    pulse = np.zeros(length)
    pulse[15:20] = 1

    time_trace = mt.shift_array(pulse, -10) + mt.shift_array(pulse, 5)
    pulses = pt.find_pulse(pulse, time_trace, 5, number_of_pulses=2)

    assert pulses[0] == length - 10 and pulses[1] == length + 5, "Pulse location inaccurate."


def test_find_peaks():
    length = 50
    pulse = np.zeros(length)
    peak = np.array([5 - abs(x) for x in range(-3, 4)])
    pulse[30:37] += peak
    pulse[40:47] += peak

    peaks, _ = pt.find_peaks(pulse)
    assert peaks[0] == 33 and peaks[1] == 43, ValueError("Peak locations inaccurate.")


def test_window():
    ones = np.ones(50).astype(np.float64)

    box_car_window = pt.window(ones, (20, 30), "box_car")
    cos_window = pt.window(ones, (20, 30), "cosine")
    tukey_window = pt.window(ones, (20, 30), "tukey", tukey_alpha=0.5)

    assert np.sum(box_car_window) == 10 and np.sum(box_car_window[20:30]) == 10, ValueError("Box_car window "
                                                                                            "has wrong shape.")
    assert h.within_tolerance(np.sum(cos_window), 6.39245, 1e-5) and \
           cos_window[np.argmax(cos_window)] == cos_window[24] == cos_window[25], ValueError("Cosine window "
                                                                                             "has wrong shape.")
    assert h.within_tolerance(np.sum(tukey_window), 6.76604, 1e-5) and \
           tukey_window[24] == tukey_window[25] == 1, ValueError("Tukey window has wrong shape.")
