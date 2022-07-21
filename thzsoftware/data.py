import os
import numpy as np
import pandas as pd
# test

def _load(data_directory, data_type):
    path = data_directory
    dirs = os.listdir(path)
    dir_keys = []
    dir_dict = {}

    # start by searching through directories (intended to be e.g. measurement days)
    for directory in dirs:
        dir_keys.extend([directory])
        dir_path = path + '/' + directory

        measurement_files = os.listdir(dir_path)
        measurement_keys = []
        measurement_dict = {}

        # go through data files
        for file in measurement_files:
            measurement_keys.extend([file])
            file_path = dir_path + '/' + file
            data_frame = pd.read_csv(file_path)

            file_key = os.path.splitext(file)[0]
            measurement_dict[file_key] = {}
            measurement_dict[file_key][data_type] = data_frame

        dir_dict[directory] = measurement_dict.copy()

    return dir_dict


class DataSet:
    def __init__(self, data_directory, data_type):
        assert isinstance(data_directory, str), TypeError("'data_directory' must be string.")
        assert isinstance(data_type, str), TypeError("'data_type' must be string. This option indicates the nature of "
                                                     "the data contained in the directory. E.g. 'spectral' or 'time "
                                                     "series'.")

        data = _load(data_directory, data_type)

        self.data = data
        self.data_path = data_directory
        self.keys = self.get_keys()

    def add_entry(self, dir_key, meas_key, type_key, data_packet):
        assert isinstance(data_packet, tuple), TypeError("data_packet should have type tuple."
                                                         f" type(data_packet) = {type(data_packet)}")
        df_dict = {}
        for col_title, col_array in data_packet:
            assert isinstance(col_title, str), TypeError("col_title must be str."
                                                         f" type(col_title) = {type(col_title)}")
            assert isinstance(col_array, (list, tuple, np.ndarray)), TypeError("col_array must be array-like."
                                                                               f" type(col_array) = {type(col_array)}")
            df_dict[col_title] = col_array

        df = pd.DataFrame(df_dict)

        if dir_key not in self.data.keys():
            self.data[dir_key] = {}
        if meas_key not in self.data[dir_key].keys():
            self.data[dir_key][meas_key] = {}
        if type_key not in self.data[dir_key][meas_key].keys():
            self.data[dir_key][meas_key][type_key] = {}

        self.data[dir_key][meas_key][type_key] = df

    def get_keys(self):
        out = {}
        for dir_key in self.data:
            out[dir_key] = {}
            for measurement_key in self.data[dir_key]:
                out[dir_key][measurement_key] = {}
                for type_key in self.data[dir_key][measurement_key]:
                    out[dir_key][measurement_key] = type_key

        return out
