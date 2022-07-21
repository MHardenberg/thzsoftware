import pandas as pd
from thzsoftware.data import DataSet


def test_dataset_obj_init():
    test_set = DataSet("./tests/test_dummy_data/dataDir/", "dummy_type")
    dir_key = list(test_set.data.keys())[0]
    measurement_key = list(test_set.data[dir_key].keys())[0]
    type_key = list(test_set.data[dir_key][measurement_key].keys())[0]

    assert dir_key == "measurements_6th_september", ValueError("Wrong directory loaded.")
    assert measurement_key == "dummy_file", ValueError("Wrong measurement loaded.")
    assert type_key == "dummy_type", ValueError("Wrong type set.")
    assert isinstance(test_set.data[dir_key][measurement_key][type_key], pd.core.frame.DataFrame), \
        TypeError("Loaded data not pandas dataframe. type(data) = " +
                  f"{test_set.data[dir_key][measurement_key][type_key]}")


def test_dataset_obj_add_entry():
    test_set = DataSet("./tests/test_dummy_data/dataDir/", "dummy_type")
    added_dir = "added_dir"
    added_meas = "added_meas"
    added_type = "added_type"
    data_packet = (("col1", [1, 2, 3, 4, 5]), ("col2", [5, 4, 3, 2, 1]))

    test_set.add_entry(added_dir, added_meas, added_type, data_packet)
    assert added_dir in test_set.data, "Added dir not in object."
    assert added_meas in test_set.data[added_dir], "Added measurement not in object."
    assert added_type in test_set.data[added_dir][added_meas], "Added type not in object."

    assert isinstance(test_set.data[added_dir][added_meas][added_type], pd.core.frame.DataFrame),\
        TypeError("Loaded data not pandas dataframe."f" type(data) = "
                  f"{test_set.data[added_dir][added_meas][added_type]}")


def test_dataset_obj_get_keys():
    test_set = DataSet("./tests/test_dummy_data/dataDir/", "dummy_type")
    keys = test_set.get_keys()

    dir_key = "measurements_6th_september"
    measurement_key = "dummy_file"
    type_key = "dummy_type"

    assert isinstance(keys, dict), TypeError("Output should be dict of keys.")
    assert dir_key in keys, "dir_key not listed."
    assert measurement_key in keys[dir_key], "measurement key not listed."
    assert type_key in keys[dir_key][measurement_key], "type key not listed."
