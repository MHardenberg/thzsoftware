import os
from time import sleep

import numpy as np
import pandas as pd


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
        assert isinstance(data_directory, str), TypeError("'data_directory' must be string. type(data_directory) = "
                                                          f"{type(data_directory)}.")
        assert isinstance(data_type, str), TypeError("'data_type' must be string. This option indicates the nature of "
                                                     "the data contained in the directory. E.g. 'spectral' or 'time "
                                                     f"series'. type(data_type) = {type(data_type)}.")

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

    def save_to_csv(self, path=None):
        assert isinstance(path, str) or path is None, TypeError(f"Target path must be str or None. type(path) = {type(path)}")
        if path is None:
            path = self.data_path + "_output"
            print(f"Target output set to {path}")

        data = self.data
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if len(os.listdir(path)) > 0:
                user_input = None
                print("Target directory is non-empty. Continuing risks overwriting data in "
                      "directories of the same name.")
                sleep(0.01)
                while user_input not in ("y", "n"):
                    user_input = input("Continue? (y/n):  ").lower()
                if user_input == 'n':
                    sleep(0.01)
                    print("Canceled. No data saved.")
                    return

        for directory in data:
            dir_path = path + '/' + directory
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            for data_set in data[directory]:
                data_path = dir_path + '/' + data_set
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                for data_type in data[directory][data_set]:
                    file_path = data_path + '/' + data_set + '_' + data_type + ".csv"
                    data[directory][data_set][data_type].to_csv(path_or_buf=file_path)
