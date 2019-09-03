def get_binned_spike_trains(cache,session_id,stim_type,time_step=1/100):
    """obtain the binned spike trains for a given session in cache and a given stimulus type
    """
    import numpy as np

    session = cache.get_session_data(session_id)
    my_units = session.units #[session.units.structure_acronym==rec_area]
    #my_units = my_units.sort_index(by='probe_vertical_position')
    my_stim = session.get_presentations_for_stimulus(stim_type)
    first_id = my_stim.index.values[0]
    first_duration = my_stim.loc[first_id, "stop_time"] - my_stim.loc[first_id, "start_time"]
    time_domain = np.arange(0.0, first_duration + time_step, time_step)
    histograms = session.presentationwise_spike_counts(bin_edges=time_domain,stimulus_presentation_ids=my_stim.index,
        unit_ids=my_units.index.values)
    T_histo = histograms.transpose()
    return T_histo,my_units

def get_binned_spike_trains_sorted(cache,session_id,stim_type,time_step=1/100):
    """obtain the binned spike trains for a given session in cache and a given stimulus type
    """
    import numpy as np

    session = cache.get_session_data(session_id)
    my_units = session.units #[session.units.structure_acronym==rec_area]
    my_units = my_units.sort_values(by=['probe_vertical_position'], ascending = False)
    my_stim = session.get_presentations_for_stimulus(stim_type)
    first_id = my_stim.index.values[0]
    first_duration = my_stim.loc[first_id, "stop_time"] - my_stim.loc[first_id, "start_time"]
    time_domain = np.arange(0.0, first_duration + time_step, time_step)
    histograms = session.presentationwise_spike_counts(bin_edges=time_domain,stimulus_presentation_ids=my_stim.index,
        unit_ids=my_units.index.values)
    T_histo = histograms.transpose()
    return T_histo,my_units

