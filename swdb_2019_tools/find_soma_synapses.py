import numpy as np
import pandas as pd
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink

def find_soma_synapses(neuron_id, radius, synapses_name = 'pni_synapses_i3', dataset_name = 'pinky100', voxel_size = [4,4,40], sql_database_uri = 'postgresql://analysis_user:connectallthethings@swdb-em-db.crjvviai1xxh.us-west-2.rds.amazonaws.com/postgres'):
    """Function for finding the synapses within some radius around the soma for a particular 
    neuron as a proxy for finding inhibitory connections
    Parameters
    ----------
    neuron_id: int
        pt_root_id of the neuron of interest
    radius: float
        radius in nm
    synapses_name: string
        synapses to query (default = 'pni_synapses_i3')
    dataset_name: string
        dataset to query (default = 'pinky100')
    voxel_size: length(3) iterator
        a list, tuple, or numpy array with the desired voxel size (default = [4,4,40])
    sql_database_uri: string
        database to load (default = 'postgresql://analysis_user:connectallthethings@swdb-em-db.crjvviai1xxh.us-west-2.rds.amazonaws.com/postgres')
    Returns
    -------
    pd.Series
        A pandas series of ids of the synapses within the specified
        radius of the some of the specified neuron
    Example
    -------
    closest_syn_ind = find_closest_synapses(neuron_id = 648518346349478248, radius = 30000)
    """
    dl = AnalysisDataLink(dataset_name=dataset_name, sqlalchemy_database_uri=sql_database_uri, verbose=False)
    auto_pre_synapses = dl.query_synapses(synapses_name, post_ids = [neuron_id])
    neuron_df = dl.query_cell_types('soma_valence_v2', cell_type_exclude_filter=['g'])
    soma_position = convert_to_nm(neuron_df[neuron_df.pt_root_id == neuron_id]['pt_position'])
    ctr_positions_nm = convert_to_nm(auto_pre_synapses['ctr_pt_position'])
    euclidean_distances = np.apply_along_axis(lambda x: np.linalg.norm(x-soma_position[0]), 1, ctr_positions_nm)
    closest_synapses = auto_pre_synapses.id[euclidean_distances<radius]
    return closest_synapses#, euclidean_distances



#The convert_to_nm function by Forrest
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