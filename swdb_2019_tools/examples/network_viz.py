from swdb_2019_tools.load_synaptic_network import load_network_data, get_edgelist, get_soma_positions_df
from swdb_2019_tools.viz_3d_network import viz_3d_network

# load data for the e-e network
cell_types_df, synapse_df = load_network_data(['e'])
edgelist = get_edgelist(synapse_df, weight='binary')
position_df = get_soma_positions_df(cell_types_df)

viz_3d_network(position_df, edgelist)
