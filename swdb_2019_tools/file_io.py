import numpy as np
import h5py
import pandas as pd


def write_df_with_array_columns(df, path, non_array_key="non_array", **kwargs):
    """ Write a table with ragged array columns to an h5 file

    Parameters
    ==========
    df : pandas.DataFrame
        table to be written
    path : str
        path to a writeable h5 file
    non_array_key : str, optional
        special key within the .h5 file that denotes all non-array columns

    """

    array_columns = [
        colname for colname in df.columns 
        if isinstance(df[colname].values[0], np.ndarray)
    ]

    array_df = df.loc[:, array_columns]
    non_array_df = df.drop(columns=array_columns)

    with h5py.File(path, "w") as fil:
        for colname in array_columns:
            col_data = array_df.pop(colname)

            breakpoints = [0]
            for ii, (_, row) in enumerate(col_data.iteritems()):
                breakpoints.append(breakpoints[ii] + len(row))

            data = np.concatenate(col_data.values)
            index = col_data.index.values

            group = fil.create_group(colname)
            group.create_dataset("data", data=data)
            group.create_dataset("index", data=index)
            group.create_dataset("breakpoints", data=breakpoints[1:-1])

    non_array_df.to_hdf(path, key=non_array_key, **kwargs)


def read_df_with_array_columns(path, non_array_key="non_array"):
    """ Read an h5 representation of a table with ragged array columns

    Parameters
    ==========
    path : str
        Path to a .h5 file that will be read
    non_array_key : str, optional
        special key within the .h5 file that denotes all non-array columns

    Returns
    =======
    pandas.DataFrame : 
        obtained table

    """

    arr_cols = {}

    with h5py.File(path, "r") as fil:
        for key in fil.keys():

            if key == non_array_key:
                continue

            breakpoints = fil[key]["breakpoints"][:]
            arr_cols[key] = np.split(fil[key]["data"][:], breakpoints)
            index = fil[key]["index"][:]

    arr = pd.DataFrame(arr_cols, index=pd.Index(index))
    non_arr = pd.read_hdf(path, key=non_array_key)
    return pd.merge(non_arr, arr, left_index=True, right_index=True)
            
