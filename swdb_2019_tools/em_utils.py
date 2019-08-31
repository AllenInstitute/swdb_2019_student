import numpy as np


def convert_to_nm(col, voxel_size=[4,4,40]):
    """useful function for converting a pandas data frame voxel position
    column to a np.array of Nx3 size in nm

    Parameters
    ----------
    col : pandas.Series or np.array
        column from a datalink dataframe query, containing lists of x,y,z coordinates in voxels
        len(col)=N entries
    voxel_size : length(3) iterator
        a list, tuple, or numpy array with the desired voxel size (default = [4,4,40])

    Returns
    -------
    np.array
        a Nx3 array of positions converted to nanometers

    Example
    -------
    df = dl.query_cell_types('soma_valence_v2')
    soma_pos_nm = convert_to_nm(df.pt_position)
    """
    return np.vstack(col.values)*voxel_size