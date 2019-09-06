import os
import h5py
import numpy as np
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

    Example
    =======
    table = pd.DataFrame({
        "a": [[1, 2, 3], [], [4, 5, 6]],
        "b": [1, 2, 3],
    })
    write_df_with_array_columns(table, "/some/file/path/data.h5")

    """

    array_columns = [
        colname for colname in df.columns 
        if isinstance(df[colname].values[0], (np.ndarray, list))
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

    Example
    =======
    table = read_df_with_array_columns("/some/file/path/data.h5")

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
            


def get_multi_session_flash_response_df_for_container(container_id, cache):

    """ Load pre-generated multi-session flash response dataframe for a container from cache location on AWS

        Parameters
        ==========
        container_id : int
            ID of container to load
        cache : cache object
            special key within the .h5 file that denotes all non-array columns

        Returns
        =======
        pandas.DataFrame :
            flash_response_df aggregated over sessions within a container
    """

    manifest = cache.experiment_table
    cre_line = manifest[manifest.container_id==container_id].cre_line.values[0]
    imaging_depth = manifest[manifest.container_id==container_id].imaging_depth.values[0]
    expt_type = cre_line.split('-')[0]+'_'+str(imaging_depth)
    save_dir = r'/home/ec2-user/SageMaker/shared/multi_session_dataframes'
    data_df = pd.DataFrame()
    timestamps_df = pd.DataFrame()
    data_suffix = str(container_id)+'_data'
    timestamps_suffix = str(container_id)+'_timestamps'
    data_path = os.path.join(save_dir, 'image_flash_response_df_'+expt_type+'_'+data_suffix+'.h5')
    timestamps_path = os.path.join(save_dir, 'image_flash_response_df_'+expt_type+'_'+timestamps_suffix+'.h5')
    # get data
    data_tmp = pd.read_hdf(data_path, key='df')
    timestamps_tmp = pd.read_hdf(timestamps_path, key='df')
    data_df = pd.concat([data_df, data_tmp])
    timestamps_df = pd.concat([timestamps_df, timestamps_tmp])
    df = data_df.merge(timestamps_df, how='right', on=['experiment_id','flash_id'])
    return df