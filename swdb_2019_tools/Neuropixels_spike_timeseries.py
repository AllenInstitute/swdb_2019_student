import numpy as np
import pandas as pd

def get_spike_timeseries_to_df(cache, sessionIDs = [], regions = []):
    """
    Purpose: retrieve spike time data from across sessions restricted by region

    Inputs:
    sessionIDs: a list of session IDs from the sessions object.
    If left empty [], the function will default to go through all the sessions and check whether any given session
    contained the regions specified in the 'regions' argument and will add the session to the list of session IDs

    regions: a list of regions (as labeled in the manual_structure_acronym column:
    ['None', 'TH', 'DG', 'CA', 'VISmma', 'MB', 'VISpm', 'VISp', 'VISl',
       'VISrl', 'VISam', 'VIS', 'VISal', 'VISmmp'])

    Returns:
    Pandas dataframe that contains:
        one row for each unit ID,
        unit's timeseries array as a list,
        the channel on which the unit was recorded,
        all of that channel's QC data,
        the channel's vertical, horizontal, and structure label,
        the probe ID that the channel belongs to,
        the mouse genotype,
        and the session ID

    Example:

    v1_spikes = get_all_timeseries_to_df(sessionIDs = sessions.index[0:3], regions = ['VIS"])

    will return the lfp spike data for every channel recorded in VIS in the first 2 sessions listed in the sessions dataframe
    """
    #Get cache and info about all the sessions
    sessions = cache.get_sessions()
    allchannelsinfo = cache.get_channels()
    allunitsinfo = cache.get_units()
    #If no session ID is passed into the list, find all sessions that contain the regions
    #and append that session ID to the sessionIDs list
    if len(sessionIDs) == 0:
        sessionIDs = []
        for i in np.arange(len(sessions.structure_acronyms)):
            sessionid = sessions.structure_acronyms.index[i]
            if any(elem in sessions.structure_acronyms[sessionid] for elem in regions):
                sessionIDs.append(sessionid)
    #Double check that the regions specified actually appear in the sessionIDs specified
    for sessionID in sessionIDs:
        for elem in regions:
            if elem not in sessions.structure_acronyms[sessionID]:
                print("Session {} does not contain region {}.".format(sessionID, elem))
    all_spikes_df = pd.DataFrame()
    #Grab channel and unit info for each region
    for sessionID in sessionIDs:
        session_info = cache.get_session_data(sessionID)
        session_channels = session_info.channels
        session_units = session_info.units
        session_spikes = {}
        session_spikes_df = pd.DataFrame()
        session_spike_times = session_info.spike_times
        for unit in session_units.peak_channel_id[session_units.structure_acronym.isin(regions)].index:
            #print('appending unit {} from session {} for area {}'.format(unit, sessionID, region))
            session_spikes[unit] = session_spike_times[unit]
        session_spikes_df['unit_id'] = session_spikes.keys()
        session_spikes_df['spike_timeseries'] = session_spikes.values()
        session_spikes_df = pd.merge(session_spikes_df,
          session_units,
          left_on = 'unit_id',
          right_on = session_units.index)
        session_spikes_df['sessionID'] = sessionID
        session_spikes_df['genotype'] = sessions.genotype[sessions.index == sessionID].unique()[0]
        all_spikes_df = all_spikes_df.append(session_spikes_df)
    return all_spikes_df