import numpy as np
import pandas as pd

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


def synapse_to_matrix(syn_df, aggfunc = np.sum, fillvalue=0):
    """function for converting a pandas dataframe of synapses to a connectivity matrix
    
    Parameters
    ----------
    syn_df : pd.DataFrame
        Dataframe of synapses pulled from query_synapses
    aggfunc : func
        function to aggregate all the synapse sizes between pairs
        (default = np.sum)
        examples) np.sum (total synapse size)
                  np.mean (avg synapse size)
                  len (number of synapses)
                  np.std (variation of synapses)
    fillvalue : Number
        what to fill in the matrix when there are no synapses (default 0)
    
    Returns
    -------
    pd.DataFrame
        a dataframe whose rows are indexed by root_id, and columns are indexed by root id,
        values are the aggregated quantification of synapses between those.

    Example
    -------
    syn_df = dl.query_synapses('pni_synapses_i3', pre_ids = [neuron_list], post_ids = [neuron_list])
    conn_df = synapse_to_matrix(syn_df)
    
    # plot the matrix as an image
    f, ax = plt.subplots()
    ax.imshow(conn_df.values)

    """
    pd.pivot_table(syn_df,
                   values = 'size',
                   index = 'pre_pt_root_id',
                   columns = 'post_pt_root_id',
                   aggfunc=len,
                   fill_value = 0)