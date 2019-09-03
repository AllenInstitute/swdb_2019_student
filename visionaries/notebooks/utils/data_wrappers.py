"""
Helpful functions for accessing data
"""
import pandas as pd
import allensdk.brain_observatory.stimulus_info as stim_info

def get_dg_response_filter_from_saskia():
    """
    Returns a responsiveness df, with two cols:
      [
        'cell_specimen_id',
        'responsive': bool
      ]
    The source is from saskia's swdb_dg_table.csv dataset.
    """
    # Read data from file 'filename.csv' 
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later) 
    data = pd.read_csv("../data/swdb_dg_table.csv")
    return data[['cell_specimen_id', 'responsive_dg']].rename(
        columns = {'responsive_dg': 'responsive'},
        inplace = False)

def get_cells(boc, cells, brain_area, depth, cell_type, stimuli = [stim_info.DRIFTING_GRATINGS]):
    """
    Get cells that match the given selectors.
    @param boc - BrainObservatoryCache
    @param cells - DataFrame fetched like so: 
        cells = boc.get_cell_specimens()
        cells = pd.DataFrame.from_records(cells)
    """
    exps = boc.get_ophys_experiments(stimuli=stimuli,
        targeted_structures = [brain_area],
        imaging_depths = [depth],
        cre_lines = [cell_type])
    # Can't find [targeted_structure, cre_line] in cell table, so have to combine w/ exps.
    exps_df = pd.DataFrame.from_dict(exps)
    if len(exps_df)== 0:
        return []
    is_in_relevant_container_mask = False * len(cells)
    for relevant_container_id in (exps_df.experiment_container_id.values):
        is_in_relevant_container_mask |= (cells.experiment_container_id == relevant_container_id)
    return cells[is_in_relevant_container_mask]

def get_filtered_cells(cells, response_filter):
    """
    @param cells - DataFrame like 
        cells = boc.get_cell_specimens()
        cells = pd.DataFrame.from_records(cells)
    @param response_filter - DataFrame with ['cell_speciment_id', 'responsive': bool]
        E.g. get_dg_response_filter_from_saskia()
    @return filtered cells_df, but just the responsive ones.
    """
    cells = cells.merge(
            right=response_filter,
            on = 'cell_specimen_id',
            how ='left')
    # The response column has some NaN values, replace them w/ False.
    cells['responsive'].fillna(False, inplace=True)
    return cells[cells['responsive']]

def get_avg_normalized_response(boc, session_id, cell_specimen_id, temporal_frequency=2.0):
    ''' generate normalized average response for each grating orientation
    '''
    data_set = boc.get_ophys_experiment_data(session_id)
    
    timestamps, dff = data_set.get_dff_traces(cell_specimen_ids=[cell_specimen_id])
    dff_trace = dff[0,:]
    
    stim_table = data_set.get_stimulus_table('drifting_gratings')
    
    #Calculate the mean DF/F for each grating presentation in this stimulus
    rows = list()
    for i in range(len(stim_table)):
        new_row = {
            'orientation': stim_table.orientation[i],
            'temporal_frequency': stim_table.temporal_frequency[i],
            'mean_dff': dff_trace[stim_table.start[i]:stim_table.end[i]].mean()
        }
        rows.append(new_row)

    cell_response = pd.DataFrame.from_dict(rows)
    tf2_response = cell_response[cell_response.temporal_frequency==temporal_frequency]
    
    mean_dff_ori = tf2_response.groupby('orientation').mean()
    mean_dff_ori = mean_dff_ori.clip(lower=pd.Series({'mean_dff': 0.0}), axis=1)
    
    max_response = mean_dff_ori['mean_dff'].max()
    
    return mean_dff_ori['mean_dff']/max_response

def convert_polar_dict_to_arrays(polar_series):
    thetas = []
    rs = []
    for theta, r in polar_series.iteritems():
        thetas.append(theta)
        rs.append(r)
    return thetas, rs
