#Function that takes in: Session ID(s) as list (sessionIDs),
    #brain regions (using manual_structure_acronym) as list (regions),
    #and datatype as 'spikes', 'lfp', or 'both' (defaults to both)
    #IF session IDs is left blank, then it will default to search through all the sessions that contain the brain regions(s)
    #and will grab all the data across all sessions in which the brain region(s) was recorded
#Spits out a PANDAS DATAFRAME for each datatype- if asking for "both", then provide a variable name for EACH dataframe:
    #LFP contains:
        #one row for each channelID,
        #the timeseries array as a list,
        #the channel's vertical, horizontal, and structure label,
        #the probe ID that the channel belongs to,
        #the mouse genotype,
        #and the session ID
    #Spikes contains:
        #one row for each unit ID,
        #unit's timeseries array as a list,
        #the channel on which the unit was recorded,
        #the channel's vertical, horizontal, and structure label,
        #the probe ID that the channel belongs to,
        #the mouse genotype,
        #and the session ID
#Import all the things
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys import ecephys_session
%matplotlib inline

# fix slow autocomplete
%config Completer.use_jedi = False

#Set path to cache
import platform
platstring = platform.platform()

if 'Darwin' in platstring:
    # OS X
    data_root = "/Volumes/Brain2019/"
elif 'Windows'  in platstring:
    # Windows (replace with the drive letter of USB drive)
    data_root = "E:/"
elif ('amzn1' in platstring):
    # then on AWS
    data_root = "/data/"
else:
    # then your own linux platform
    # EDIT location where you mounted hard drive
    data_root = "/media/$USERNAME/Brain2019/"

manifest_path = os.path.join(data_root, "dynamic-brain-workshop/visual_coding_neuropixels/2019/manifest.json")

#Get cache and info about all the sessions
cache = EcephysProjectCache.fixed(manifest=manifest_path)
sessions = cache.get_sessions()
allchannelsinfo = cache.get_channels()
allunitsinfo = cache.get_units()


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
        for region in regions:
            region_lfps = {}
            region_spikes = {}
            region_lfps_df = pd.DataFrame()
            region_spikes_df = pd.DataFrame()
            region_channelinfo = session_channels[session_channels.manual_structure_acronym == region]
            region_units = session_units[session_units.peak_channel_id.isin(region_channelinfo.index)]
            #Depending on what type of data, grab the lfps and/or spike times
            if datatype == "both":
                session_spike_times = session_info.spike_times
                for probeid in session_channels.probe_id[session_channels.manual_structure_acronym == region].unique():
                    print('retrieving probe {} from session {} cache'.format(probeid, sessionID))
                    probe_lfp = session_info.get_lfp(probeid)
                    region_channels_lfp = probe_lfp.loc[dict(channel = probe_lfp.channel.isin(region_channelinfo.index))]
                    for chan in region_channels_lfp["channel"].values:
                        print('appending channel {} from probe {} for area {}'.format(chan, probeid, region))
                        region_lfps[chan] = region_channels_lfp[:, 'channel' == chan].values
                for unit in region_units.index:
                    print('appending unit {} from session {} for area {}'.format(unit, sessionID, region))
                    region_spikes[unit] = session_spike_times[unit]
            elif datatype == "lfp":
                for probeid in session_channels.probe_id[session_channels.manual_structure_acronym == region].unique():
                    print('retrieving probe {} from session {} cache'.format(probeid, sessionID))
                    probe_lfp = session_info.get_lfp(probeid)
                    region_channels_lfp = probe_lfp.loc[dict(channel = probe_lfp.channel.isin(region_channelinfo.index))]
                    for chan in region_channels_lfp["channel"].values:
                        print('appending channel {} from probe {} for area {}'.format(chan, probeid, region))
                        region_lfps[chan] = region_channels_lfp[:, 'channel' == chan].values
            elif datatype == "spikes":
                session_spike_times = session_info.spike_times
                for unit in region_units.index:
                    print('appending unit {} from session {} for area {}'.format(unit, sessionID, region))
                    region_spikes[unit] = session_spike_times[unit]
            #Build up the dataset after each region is added
            #lfp dataset
            print('putting all the lfps from region {} and session {} to the larger dataset if lfps were requested'.format(region, sessionID))
            region_lfps_df['channel_id'] = region_lfps.keys()
            region_lfps_df['lfp_timeseries'] = region_lfps.values()
            region_lfps_df = pd.merge(region_lfps_df,
              region_channelinfo.loc[:, 'manual_structure_acronym':'probe_id'],
              left_on = 'channel_id',
              right_on = region_channelinfo.index)
            region_lfps_df['sessionID'] = sessionID
            region_lfps_df['genotype'] = sessions.genotype[sessions.index == sessionID].unique()[0]
            #spike dataset
            print('putting all the spikes from region {} and session {} to the larger dataset if spikes were requested'.format(region, sessionID))
            region_spikes_df['unit_id'] = region_spikes.keys()
            region_spikes_df['spike_timeseries'] = region_spikes.values()
            region_spikes_df = pd.merge(region_spikes_df,
              region_units,
              left_on = 'unit_id',
              right_on = region_units.index)
            region_spikes_df['sessionID'] = sessionID
            region_spikes_df['genotype'] = sessions.genotype[sessions.index == sessionID].unique()[0]
            #append to the overarching datasets that aren't tied to specific session id or region
            all_lfps_df = all_lfps_df.append(region_lfps_df)
            all_spikes_df = all_spikes_df.append(region_spikes_df)
    if datatype == "both":
        return all_lfps_df, all_spikes_df
    elif datatype == "lfp":
        return all_lfps_df
    elif datatype == "spikes":
        return all_spikes_df
