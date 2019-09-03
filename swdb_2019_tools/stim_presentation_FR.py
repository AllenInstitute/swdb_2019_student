import pandas as pd

def stim_presentation_FR(session, stimulus): #session,stimulus of interest



    #get session stimulus presentation info for stimulus of interest, create index
    stim_table = session.get_presentations_for_stimulus(stimulus)
    stim_ids = stim_table.index.values 

    #define the duration of each stimulus presentation 
    durations = stim_table["duration"]

    #get spike count per unit for length of stimulus presentation
    spikes_per_stim = session.presentationwise_spike_times(stim_ids)\
        .reset_index()\
        .groupby(["stimulus_presentation_id", "unit_id"]).count()\
        .rename(columns={"spike_time": "spike_count"})

    #help(session.presentationwise_spike_counts) #documentation

    #merge spike counts and length of stimulus 
    spikes_per_stim = spikes_per_stim.reset_index()
    durations = session.stimulus_presentations["duration"]
    spikes_per_stim = pd.merge(spikes_per_stim, durations, left_on="stimulus_presentation_id", right_on="stimulus_presentation_id")


    #calculate FR of each unit
    stim_FR = (spikes_per_stim.spike_count/spikes_per_stim.duration)
    return stim_FR   