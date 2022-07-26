import numpy as np
from scipy import signal
from thzsoftware import math as mt
from thzsoftware import fitting as fit


# convolve mask array over base array
def convolve(base, mask, dx=1):
    assert len(base) == len(mask), "Length of input array must be equal."
    base = base.astype(np.float64)
    mask = mask.astype(np.float64)  # the 'base' is which ever array is held static, while the mask is
    # being shifted over it.  Typically, one would use a reference pulse as mask, shifting it over a sample base,
    # to determine delay between two equally shaped pulses.

    xs = dx * np.array(range(-len(base), len(base)))[::-1]
    conv = np.empty_like(xs).astype(np.float64)
    counter = 0

    # For simplicity, we approximate the integral with rectangular columns.
    for i in range(-len(base), len(base)):
        mask_shifted = mt.shift_array(mask, i)
        conv[counter] = np.sum(dx * base * mask_shifted)
        counter += 1
    return xs, conv


# use convolution method to find the highest degrees of overlap between a pulse and a time trace. Multiple pulse can be
# found, when e.g. looking for echos.
def find_pulse(pulse, time_trace, pulse_width_indices, number_of_pulses=1):
    _, conv = convolve(time_trace, pulse)
    peaks, _ = find_peaks(conv)
    peaks = peaks[np.argsort(conv[peaks])]  # sort peaks by size.

    out = []
    counter = 0
    while len(out) < number_of_pulses:  # for number of pulses, we find the highest, and then discard any of the
        # following, which are to close to it. These are assumed to be part of the same peak.
        peak = peaks[counter]
        within_pulse = False
        for earlier_peak in out:
            if (earlier_peak - pulse_width_indices / 2) < peak < (earlier_peak + pulse_width_indices / 2):
                within_pulse = True
        if not within_pulse:
            out.extend([peak])
        counter += 1
    return peaks


def find_peaks(pulse):
    # to be expanded. Would like some stability against noise.
    # output = (peaks, properties)
    return signal.find_peaks(pulse)


def define_thz_pulses(ys, number_of_pulses=1):
    assert isinstance(number_of_pulses, (int, np.integer)), TypeError("number_of_pulses kwarg must be of type integer."
                                                                      " type(number_of_pulses) = "
                                                                      f"{type(number_of_pulses)}")
    assert isinstance(ys, (tuple, list, np.ndarray)), \
        TypeError("Input array must be array-like.", f" type(ys) = {type(ys)}")

    # This function assumes a pulse shape of a smaller negative peak, followed by a dominant positive peak.
    # We invert the function should the signs be opposite. This does not affect returned values.
    if np.max(ys) < -np.min(ys):
        ys = -ys
        # If most extreme peak is negative invert trace to keep calculations simple.

    assert np.max(np.abs(ys)) != 0., ValueError("Time trace has 0 field in all points.")
    assert np.min(ys) < 0, ValueError("Time trace for THz pulse should cross 0.")

    found_pulses = []
    peaks, _ = find_peaks(ys)
    peaks = list(peaks[np.argsort(ys[peaks])])[::-1]  # sort peaks by size.

    zero_crossings = mt.zero_crossings(ys)  # for defining the limits
    pulse_counter = 0
    peak_counter = 0
    left_limit, right_limit = 0, 0

    while peaks and pulse_counter < number_of_pulses:
        peak = peaks[peak_counter]
        peak_counter += 1

        if peak not in found_pulses:
            found_pulses.extend([peak])
            pulse_counter += 1

            # find left limit.
            #   Here we assume, that we are interested in a region extending past the dominant peak, as far as the
            #   minimum is removed from the peak. This is to capture the slow tapering to zero on the left-hand side
            #   of the pulse
            left_crossing_minus_1, left_crossing = zero_crossings[zero_crossings < peak][-2:]
            left_min = np.argmin(ys[left_crossing_minus_1:left_crossing]) + left_crossing_minus_1

            dist_to_min = peak - left_min
            left_limit = peak - 2 * dist_to_min - (peak - left_min) // 4

            # find right limit
            #   Here we set to right-hand limit to the second zero crossing after the dominant pulse.
            # to avoid taking double crossings, we add the rule, that it must be more than half the peak width removed,
            # from the first crossing after the dominant peak.

            right_limit = zero_crossings[zero_crossings > peak][0]
            right_limit = zero_crossings[zero_crossings > (right_limit + 4)][0]

        # make sure the indices are valid:
        left_limit = max(0, left_limit)
        right_limit = max(0, right_limit)

    return left_limit, right_limit


# apply window function
def window(ys, limits, window_func="tukey", tukey_alpha=.1):
    assert isinstance(ys, (tuple, list, np.ndarray)), \
        TypeError("Input array must be array-like.", f" type(ys) = {type(ys)}")
    assert isinstance(limits, (tuple, list, np.ndarray)) and len(limits) == 2, \
        TypeError("Input limits must be array-like of length 2.", f" type(limits) = {type(limits)},"
                                                                  f" len(limits) = {len(limits)}.")

    if limits[0] > limits[1]:
        limits = tuple(list(limits).sort())

    assert limits[0] >= 0 and limits[1] < len(ys), ValueError(f"limits out of bound. limits = {limits} for array of "
                                                              f"len(ys) = {len(ys)}.")
    window_func = window_func.lower()
    permitted_windows = ("boxcar", "box_car", "tukey", "cosine")
    assert window_func in permitted_windows, ValueError("Window function not permitted. Permitted functions are:  "
                                                        " ".join(permitted_windows))

    window_array = np.zeros(len(ys))

    points = limits[1] - limits[0]

    if window_func == "boxcar" or window_func == "box_car":
        window_array[limits[0]:limits[1]] = 1
        return ys * window_array

    if window_func == "cosine":
        window_array[limits[0]:limits[1]] = signal.cosine(points)
        return ys * window_array

    if window_func == "tukey":
        window_array[limits[0]:limits[1]] = signal.tukey(points)
        return ys * window_array

    raise Exception("No switch statement triggerd. Something went terribly. "
                    "Please contact your local medium or spiritual guide.")
