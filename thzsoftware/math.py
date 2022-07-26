import numpy as np


# shift array by num and fill with zeros
def shift_array(array, shift, fill_value=0):
    assert isinstance(array, np.ndarray), TypeError("Input 'array' must be type numpy.ndarray."
                                                    f" type(array) = {type(array)}")
    assert isinstance(shift, (int, np.integer)), TypeError("Input 'shift' must be type int or numpy.integer."
                                                           f" type(shift) = {type(shift)}")
    assert isinstance(fill_value, (int, float, np.integer, np.floating)), \
        TypeError("Input 'fill_value' must be int, float, numpy.integer or numpy.floating.",
                  f" type(fill_value) = {type(fill_value)}")

    result = np.empty_like(array)
    if shift > 0:
        result[:shift] = fill_value
        result[shift:] = array[:-shift]
    elif shift < 0:
        result[shift:] = fill_value
        result[:shift] = array[-shift:]
    else:
        result[:] = array
    return result


# polar form of complex vector:
def carthesian_to_polar(complex_array, tolerance=1e-16):
    assert isinstance(complex_array, np.ndarray), TypeError("'complex_array' must be numpy.array.")
    assert isinstance(tolerance, (int, float)), TypeError("'tolerance' kwarg must be int of float. type(tolerance) = "
                                                          f"{type(tolerance)}")

    c_arr = complex_array.copy()  # avoid messing with array outside function
    r = np.abs(c_arr)
    # There is a complexity issue, where the imaginary part can actually be set to zero within the numpy function.
    # to avoid this we find the poles:
    poles = np.where(np.abs(c_arr.imag) < tolerance)
    c_arr.imag[poles] = 1  # will be corrected by setting phase to pi/2

    phi = np.arctan(c_arr.imag / c_arr.real)
    phi[poles] = .5 * np.pi * np.sign(c_arr.real[poles] * c_arr.imag[poles])  # correct the poles

    return r, phi


def unwrap_angles(angles):  # by convention, we assume decreasing phase
    assert isinstance(angles, (list, tuple, np.ndarray)), TypeError("Input must be array-like. type(angles) "
                                                                    f"= {type(angles)}")
    if angles.dtype == np.int32 or angles.dtype == np.int64:
        angles = angles.astype(np.float64)

    if angles[0] > 0:
        angles -= 2 * np.pi

    for i in range(1, len(angles)):
        while angles[i] > angles[i - 1]:
            angles[i] -= 2 * np.pi
    return angles


def zero_crossings(input_array, find_index="before"):
    find_index = find_index.lower()
    options = ("before", "after")

    assert isinstance(input_array, (list, tuple, np.ndarray)), TypeError("Input array must be array like.")
    if not isinstance(input_array, np.ndarray):
        input_array = np.array(input_array)
    assert find_index in options, ValueError("'find_index' option invalid. Use any of: ", ", ".join(options))
    if find_index == "before":
        out = np.where(np.diff(np.sign(input_array)))[0]
    elif find_index == "after":
        out = np.where(np.diff(np.sign(input_array)))[0] + 1
    return out
