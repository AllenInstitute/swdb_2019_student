def get_all_timeseries_to_df(sessionIDs = [], regions = [], datatype = "both"):
    """
    Purpose: retrieve LFP and/or spike time data from across sessions restricted by region

    Inputs:
    sessionIDs: a list of session IDs from the sessions object.
    If left empty [], the function will default to go through all the sessions and check whether any given session
    contained the regions specified in the 'regions' argument and will add the session to the list of session IDs

    regions: a list of regions (as labeled in the manual_structure_acronym column:
    ['None', 'TH', 'DG', 'CA', 'VISmma', 'MB', 'VISpm', 'VISp', 'VISl',
       'VISrl', 'VISam', 'VIS', 'VISal', 'VISmmp'])

    datatype: "lfps", "spikes", or "both". Defines whether you want to retrieve just the lfp data for the sessionIDs and
    regions, just the spike data, or both. Defaults to both

    Returns:
    Pandas dataframe for each datatype- if datatype is "both", then provide a variable name for EACH dataframe:
    LFP dataset contains:
        one row for each channelID,
        the timeseries array as a list,
        the channel's vertical, horizontal, and structure label,
        the probe ID that the channel belongs to,
        the mouse genotype,
        and the session ID
    Spikes contains:
        one row for each unit ID,
        unit's timeseries array as a list,
        the channel on which the unit was recorded,
        all of that channel's QC data,
        the channel's vertical, horizontal, and structure label,
        the probe ID that the channel belongs to,
        the mouse genotype,
        and the session ID

    Example:

    v1_lfps, v1_spikes = get_all_timeseries_to_df(sessionIDs = sessions.index[0:3], regions = ['VISp'], datatype = "both")

    will return the lfp and spike data for every channel recorded in VISp in the first 2 sessions listed in the sessions dataframe
    """
    #Get cache and info about all the sessions
    cache = EcephysProjectCache.fixed(manifest=manifest_path)
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
    all_lfps_df = pd.DataFrame()
    all_spikes_df = pd.DataFrame()
    #Grab channel and unit info for each region
    for sessionID in sessionIDs:
        session_info = cache.get_session_data(sessionID)
        session_channels = session_info.channels
        session_probes = session_info.probes
        session_units = session_info.units
        if datatype == 'both':
            session_lfps_df = pd.DataFrame()
            session_spikes_df = pd.DataFrame()
            session_lfps = {}
            session_spikes = {}
            session_spike_times = session_info.spike_times
            for probeid in session_probes.index:
                if any(session_channels.manual_structure_acronym[session_channels.probe_id == probeid].isin(regions))
                    probe_lfps = session_info.get_lfp(probeid)
                    probe_lfps_df = pd.DataFrame()
                probe_lfps_df['channel_id'] = probe_lfps[:, 'channel']
                probe_lfps_df['lfp_timeseries'] = probe_lfps[:, 'channel'].values
                probe_lfps_df = pd.merge(probe_lfps_df,
                          session_channels.loc[:, 'manual_structure_acronym':'probe_id'],
                          left_on = 'channel_id',
                          right_on = session_channels.index)
                probe_lfps_df['sessionID'] = sessionID
                probe_lfps_df['genotype'] = sessions.genotype[sessions.index == sessionID].unique()[0]
                probe_lfps_df = probe_lfps_df[probe_lfps_df.manual_structure_acronym.isin(regions)]
            all_lfps_df = all_lfps_df.append(probe_lfps_df)
            for unit in session_units.peak_channel_id[session_units.structure_acronym.isin(regions)].index:
                print('appending unit {} from session {} for area {}'.format(unit, sessionID, region))
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
        elif datatype == 'lfp':
            session_lfps = {}
            session_lfps_df = pd.DataFrame()
            for probeid in session_probes.index:
                probechannels = []
                for region in regions:
                    probechannels.append(session_channels.index[(session_channels.probe_id == probeid) & (session_channels.manual_structure_acronym == region)])
                flat_probechannels = [item for sublist in probechannels for item in sublist]
                if len(flat_probechannels) > 0:
                    print('channel info for the brain regions you specified were found on probe {} for session {}. Retreiving lfps from probe'.format(probeid, sessionID))
                    probe_lfps = session_info.get_lfp(probeid)
                    for chan in flat_probechannels:
                        chan_lfps = probe_lfps[:, 'channel' == chan].values
                        if len(session_lfps) == 0:
                            print('appending first channel ({}) from session {}'.format(chan, sessionID))
                            session_lfps[chan] = chan_lfps
                            chan_minus_1 = chan
                        elif abs((chan_lfps[0:1000] - session_lfps[chan_minus_1][0:1000])).max() > 0:
                            print('channel {} contained unique lfp information for session {}. appending now'.format(chan, sessionID))
                            session_lfps[chan] = chan_lfps
                            chan_minus_1 = chan
                        elif abs((chan_lfps[0:1000] - session_lfps[chan_minus_1][0:1000])).max() == 0:
                                  print('no unique lfp information on channel {}, not adding to dataset for session {}'.format(chan, sessionID))
            session_lfps_df['channel_id'] = session_lfps.keys()
            session_lfps_df['lfp_timeseries'] = session_lfps.values()
            session_lfps_df = pd.merge(session_lfps_df,
                          session_channels.loc[:, 'manual_structure_acronym':'probe_id'],
                          left_on = 'channel_id',
                          right_on = session_channels.index)
            session_lfps_df['sessionID'] = sessionID
            session_lfps_df['genotype'] = sessions.genotype[sessions.index == sessionID].unique()[0]
            all_lfps_df = all_lfps_df.append(session_lfps_df)            
        elif datatype == 'spikes':
            session_spikes = {}
            session_spikes_df = pd.DataFrame()
            session_spike_times = session_info.spike_times
            for unit in session_units.peak_channel_id[session_units.structure_acronym.isin(regions)].index:
                print('appending unit {} from session {} for area {}'.format(unit, sessionID, region))
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
    if datatype == "both":
        return all_lfps_df, all_spikes_df
    elif datatype == "lfp":
        return all_lfps_df
    elif datatype == "spikes":
        return all_spikes_df
