import pandas as pd
from warnings import warn
import numpy as np

from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink


def load_network_data(cell_filter=['e', 'i']):
    """
    Loads the cell_types_df and synapse_df from database

    Parameters
    ----------
    cell_filter: list
        Which cell types to include. Must be list containing ['e', 'i', 'g'].

    Output
    ------
    cell_types_df, synapse_df

    cell_types_df: pd.DataFrame, (n_neurons, )
        Information about each cell.

    synapse_df: pd.DataFrame, (n_synapses, )
        Information about each synapse.
    """

    dataset_name = 'pinky100'
    sql_database_uri = 'postgresql://analysis_user:connectallthethings@swdb-em-db.crjvviai1xxh.us-west-2.rds.amazonaws.com/postgres'

    dl = AnalysisDataLink(dataset_name=dataset_name,
                          sqlalchemy_database_uri=sql_database_uri,
                          verbose=False)

    # get all excitatory neurons
    cell_types_df = dl.query_cell_types('soma_valence_v2',
                                        cell_type_include_filter=cell_filter)

    cell_types_df = cell_types_df.set_index('pt_root_id')

    # get the edge list
    synapse_df = dl.query_synapses('pni_synapses_i3',
                                   pre_ids=cell_types_df.index,
                                   post_ids=cell_types_df.index)

    # only keep valid observations
    cell_types_df = cell_types_df.loc[cell_types_df['valid'], :]
    synapse_df = synapse_df.loc[synapse_df['valid'], :]

    return cell_types_df, synapse_df


def get_edgelist(synapse_df, weight='binary'):
    """
    Returns the edgelist.

    Parameters
    ----------
    weight: str
        What kind of edges ['binary', 'count', 'size', 'avg_size'].

    Output
    ------
    pd.DataFrame, (n_nodes, 2)
    """
    assert weight in ['binary', 'count', 'size', 'avg_size']
    edgelist = synapse_df[['pre_pt_root_id', 'post_pt_root_id']]

    if weight == 'avg_size':
        warn('SOMETHING CRAZY IS HAPPENING. Turning this edgelist'
             'into adj mat is not working....')

    if weight in ['count', 'binary']:
        edgelist = synapse_df[['pre_pt_root_id',
                               'post_pt_root_id']]
        edgelist = edgelist.groupby(['pre_pt_root_id',
                                     'post_pt_root_id']).size().reset_index()
        edgelist = edgelist.rename(columns={0: 'weight'})

    elif weight in ['size', 'avg_size']:
        edgelist = synapse_df[['pre_pt_root_id', 'post_pt_root_id', 'size']]
        edgelist = edgelist.rename(columns={'size': 'weight'})

    if weight == 'binary':
        edgelist = edgelist.drop(columns='weight')

    if weight == 'avg_size':
        edgelist = edgelist.groupby(['pre_pt_root_id',
                                     'post_pt_root_id']).mean().reset_index()

    return edgelist


def get_soma_positions_df(node_metadata, to_nm=True):
    """
    Gets the soma positions as a pd.DataFrame. Converts units to nm.

    Parameters
    ----------
    node_metadata: pd.DataFrame

    Output
    ------
    pd.DataFrame
    """
    positions = np.array(list(node_metadata['pt_position'].values))

    if to_nm:
        positions = np.multiply(positions, [4, 4, 40])

    positions = pd.DataFrame(positions,
                             index=node_metadata.index,
                             columns=['x', 'y', 'z'])

    return positions
