import os

import pytest
import pandas as pd
import numpy as np

from swdb_2019_tools import file_io


@pytest.mark.parametrize("df", [
    pd.DataFrame({
        "a": [np.arange(10), np.array([], dtype=int), np.arange(2)],
        "b": [1, 2, 3],
        "c": ["a", "B", "0"]
    }),
    pd.DataFrame({
        "a": [np.arange(10), np.arange(4)],
        "b": [1, 2],
        "c": ["a", "B"]
    }, index=pd.Index([10, -2])),
    pd.DataFrame({
        "a": [np.arange(np.random.randint(0, 10000)) for ii in range(100)],
    }),
    pd.DataFrame({
        "a": [np.arange(25).reshape((5, 5)), np.eye(5, dtype=int)],
    }),
    pd.DataFrame({
        "a": [[1, 2, 3], [], [4, 5, 6]],
        "b": [1, 2, 3],
    })
])
def test_write_df_with_array_columns(tmpdir_factory, df):

    tmpdir = str(tmpdir_factory.mktemp("arr_col"))
    path = os.path.join(tmpdir, "test.h5")

    file_io.write_df_with_array_columns(df, path)
    obt = file_io.read_df_with_array_columns(path)

    pd.testing.assert_frame_equal(df, obt, check_like=True)