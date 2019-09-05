"""
Helpful functions for accessing data
"""
import pandas as pd
import allensdk.brain_observatory.stimulus_info as stim_info
from scipy.stats import pearsonr
import numpy as np
import math

def get_dg_response_filter_from_saskia():
    """
    Returns a responsiveness df, with two cols:
      [
        'cell_specimen_id',
        'responsive': bool
        'pref_dir',
        'pref_tf'
      ]
    The source is from saskia's swdb_dg_table.csv dataset.
    """
    # Read data from file 'filename.csv' 
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later) 
    data = pd.read_csv("../data/swdb_dg_table.csv")
    return data[['cell_specimen_id', 'responsive_dg', 'pref_dir_dg','pref_tf_dg']].rename(
        columns = {'responsive_dg': 'responsive', 'pref_dir_dg': 'pref_dir', 'pref_tf_dg': 'pref_tf'},
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
    @return None if the max response is not positive. Some cells have negative mean dff values for all directions. 
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
    # Clip negative values. Some examples: cell 517410416, exp 501271265, ec 511509529
    # Talked to shawn and marina, and verified that these negative values are pretty small, so we should be good.
    mean_dff_ori = mean_dff_ori.clip(lower=pd.Series({'mean_dff': 0.0}), axis=1)
    #if min_mean_dff < 0:
    #  print ("Ignoring cell", cell_specimen_id, "because min_mean_dff =", min_mean_dff)
    #  return None

    max_response = mean_dff_ori['mean_dff'].max()
    if max_response == 0:
      return None
    
    return mean_dff_ori['mean_dff']/max_response

def convert_polar_dict_to_arrays(polar_series):
    thetas = []
    rs = []
    for theta, r in polar_series.iteritems():
        thetas.append(theta)
        rs.append(r)
    return thetas, rs

def corr_one_exp(data_set, events, c1, c2, use_events, noise_corr_else_avg_temp_corr):
    """Calculate one of [noise correlatio, trial-avg temporal corr] for movie presos for one experiment.
    @param use_events if True, use events, else use dff
    @param do_noise_corr if True, get all the presos, then do noise correlation,
    else, average out temporal correlation for each preso.

    Note about spontaneous:
    I checked that sesh A and B just have 1 spontaenous preso, session C have 2.
    So, only the big grey chunks are counted. The small ones aren't.

    Note about movies:
    Each row in stim table for the movie is a frame of a particular trial
    Trial number = repeat. 
    I'm just using natural movie 1 because it exists in all experiments.
    See http://alleninstitute.github.io/AllenSDK/_static/container_session_layout.png
    """

    try: 
        cidxs = data_set.get_cell_specimen_indices([c1, c2])
    except Exception as inst:
        return None

    if use_events:
      cidx1 = cidxs[0]
      cidx2 = cidxs[1]
      events1 = events[cidx1,:]
      events2 = events[cidx2,:]
    else: 
      timestamps, dff = data_set.get_dff_traces(cell_specimen_ids=[c1])
      events1 = dff[0,:]
      timestamps, dff = data_set.get_dff_traces(cell_specimen_ids=[c2])
      events2 = dff[0,:]
    """
    'natural_movie_one',
    'natural_movie_three',
    'natural_movie_two',
    'spontaneous'
    """
    stim_table = data_set.get_stimulus_table('natural_movie_one') 
    if noise_corr_else_avg_temp_corr:
      # Noise correlation
      c1_trial_results = []
      c2_trial_results = []
      for trial_i in range(stim_table.repeat.max() + 1):
          start = stim_table[stim_table.repeat==trial_i].start.min()
          end = stim_table[stim_table.repeat==trial_i].start.max()
          c1_trial_result = events1[start:end].mean()
          c2_trial_result = events2[start:end].mean()
          c1_trial_results.append(c1_trial_result)
          c2_trial_results.append(c2_trial_result)
      # Subtract mean from the trial results before correlating, following Ko 2011
      """
      Noise correlation was found by subtracting the average response from the responses to each trial,
      and then calculating the correlation coefficient of mean-subtracted responses
      """
      c1_trial_results = np.array(c1_trial_results) - np.mean(c1_trial_results)
      c2_trial_results = np.array(c2_trial_results) - np.mean(c2_trial_results)
      corr, p_value = pearsonr(c1_trial_results, c2_trial_results)
      return corr
    else:
      # do avg of temporal correlation
      # Each item is a temporal correlation of a single spontaneous presentation
      temp_corr_lists = []
      for trial_i in range(stim_table.repeat.max() + 1):
          start = stim_table[stim_table.repeat==trial_i].start.min()
          end = stim_table[stim_table.repeat==trial_i].start.max()
          ts1 = events1[start:end]
          ts2 = events2[start:end]
          temp_corr, p_value = pearsonr(ts1, ts2)
          if math.isnan(temp_corr):
              continue
          temp_corr_lists.append(temp_corr)
      if len(temp_corr_lists) == 0:
          return None
      return np.mean(temp_corr_lists)

def pairwise_dir_avg_temp_corr_one_exp(boc, ecid, eid, d1, d2, c_df, use_events, noise_corr_else_avg_temp_corr):
  """On one experiment, average temporal correlation between cell groups that prefer d1 vs d2
  Conceptually, the average correlation of spontaneous activity of a cell that likes d1 vs cell that likes d2.
  @param d1, d2 = the two directions to compare. E.g. 180.0
  @param c_df A dataframe with: [cell_specimen_id, experiment_container_id, pref_dir].
      You should already filter out the non-responsive / selective cells.
  """
  data_set = boc.get_ophys_experiment_data(eid)
  events = boc.get_ophys_experiment_events(ophys_experiment_id=eid)
  c_df = c_df[c_df.experiment_container_id == ecid]
  cs_d1 = c_df[c_df.pref_dir == d1]
  cs_d2 = c_df[c_df.pref_dir == d2]

  result = []
  for c1 in cs_d1.cell_specimen_id:
      for c2 in cs_d2.cell_specimen_id:
          if c1 == c2:
              continue
          pair_corr = corr_one_exp(data_set, events, c1, c2, use_events, noise_corr_else_avg_temp_corr)
          if pair_corr is None:
              continue
          result.append(pair_corr)
  if len(result) is 0:
      return None, None, None, None
  return np.mean(result), len(result), len(cs_d1), len(cs_d2)

def get_cell_locations(data_set):
    rois = data_set.get_roi_mask_array()
    num_cells = rois.shape[0]
    loc_x = np.zeros((num_cells))
    loc_y = np.zeros((num_cells))

    for i in range(num_cells):
        ind = np.where(rois[i])
        loc_x[i] = np.mean(ind[1])
        loc_y[i] = np.mean(ind[0])
    return loc_x, loc_y

def get_cell_distance(data_set, loc_x, loc_y, c1, c2):
    """
    @param loc's - See get_cell_locations.
    @param c1, c2 - cell specimen ids
    @return the cell distance in pixel distance.
    Note: Each 512 pixel field of view is 400 um. For Ko 2011, we want 50 um, so use 64 pixel distance.
    """
    try: 
        cidxs = data_set.get_cell_specimen_indices([c1, c2])
    except Exception as inst:
        return None
    x1,y1 = loc_x[cidxs[0]], loc_y[cidxs[0]]
    x2,y2 = loc_x[cidxs[1]], loc_y[cidxs[1]]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    