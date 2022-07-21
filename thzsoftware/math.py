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
    assert isinstance(tolerance, (int, float)), TypeError("'Tol' kwarg must be int of float.")

    r = np.abs(complex_array)
    # There is a complexity issue, where the imaginary part can actually be set to zero within the numpy function.
    # to avoid this we find the poles:
    poles = np.where(np.abs(complex_array.imag) < tolerance)
    complex_array.imag[poles] = 1  # will be corrected by setting phase to pi/2

    phi = np.arctan(complex_array.imag / complex_array.real)
    phi[poles] = .5 * np.pi * np.sign(complex_array.real[poles] * complex_array.imag[poles])  # correct the poles

    return r, phi


def unwrap_angles(angles):  # by convention, we assume decreasing phase
    reduce = 0
    out = np.zeros(angles.shape)
    counter = 1
    if angles[0] > 0:
        angles -= reduce

    previous = angles[0]
    out[0] = previous

    for angle in angles[1:]:
        angle -= reduce  # this is equivalent to reducing angles[n:] by 2*pi for every shift
        while angle > previous:
            angle -= 2 * np.pi
            reduce += 2 * np.pi

        previous = angle
        out[counter] = angle
        counter += 1
    return out


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

