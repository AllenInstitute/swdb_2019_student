print("Really can't wait to get my te matrix")
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
# from allensdk.brain_observatory.ecephys import ecephys_session

# import sys
# testdir = os.getcwd() 
# import smite

# import platform
# platstring = platform.platform()

# if 'Darwin' in platstring:
#     # OS X 
#     data_root = "/Volumes/Brain2019/"
# elif 'Windows'  in platstring:
#     # Windows (replace with the drive letter of USB drive)
#     data_root = "E:/"
# elif ('amzn1' in platstring):
#     # then on AWS
#     data_root = "/data/"
# else:
#     # then your own linux platform
#     # EDIT location where you mounted hard drive
#     data_root = "/media/$USERNAME/Brain2019/"
#     data_root = "/run/media/tom.chartrand/Brain2019"

# manifest_path = os.path.join(data_root, "dynamic-brain-workshop/visual_coding_neuropixels/2019/manifest.json")

# cache = EcephysProjectCache.fixed(manifest=manifest_path)

###########################################################

def save_object(obj, filename):
     with open(filename, 'wb') as output:  
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

###########################################################

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
    my_units = my_units.sort_values(by=['structure_acronym', 'probe_vertical_position'], ascending = False)

    my_stim = session.get_presentations_for_stimulus(stim_type)
    first_id = my_stim.index.values[0]
    first_duration = my_stim.loc[first_id, "stop_time"] - my_stim.loc[first_id, "start_time"]
    time_domain = np.arange(0.0, first_duration + time_step, time_step)
    histograms = session.presentationwise_spike_counts(bin_edges=time_domain,stimulus_presentation_ids=my_stim.index,
        unit_ids=my_units.index.values)
    T_histo = histograms.transpose()
    return T_histo,my_units



def get_te_matrix(session_id, save_path = os.getcwd()):

    ''' will insert description later '''

# Sets stimulus type as drifting gratings, and defines the areas to be compared
# Get binned spike trains for all units

    stim_type='drifting_gratings'
    time_step = 1/100

    areas1 = ['VISp']
    areas2 = ['VISp','VISl','VISl','VISal','VISrl','VISam','VISpm']

    T,units = get_binned_spike_trains_sorted(cache,session_id,stim_type,time_step)

#select specific stimuli - leaves option to use natural scenes but need to hack away
#at "unique stimuli" below


    print("getting session data and presentationwise spike counts...")


    session = cache.get_session_data(session_id)
    stim_table = session.get_presentations_for_stimulus(stim_type)
    stim_ids = stim_table.index.values
    if stim_type == 'natural_scenes':
        print('natural scenes')
        stim_cond = stim_conditions = stim_table.stimulus_condition_id.values
        unique_cond = stim_table.stimulus_condition_id.unique()
        frames = session.get_stimulus_parameter_values(stimulus_presentation_ids=stim_ids, drop_nulls=False)
    elif stim_type == 'drifting_gratings':
        print('drifting')
        unique_cond = stim_table.orientation.unique()
        unique_cond = unique_cond[unique_cond !='null']
        stim_cond = stim_conditions = stim_table.orientation.values

# for each stimulus orientation average across trials 

    num_stims = len(unique_cond)
    Avg_psth = np.zeros((T.shape[0],T.shape[1],num_stims))
    for ix,cond in enumerate(unique_cond):
        #print(ix)
        idx_in = stim_cond == cond
        #print(sum(idx_in))
        Avg_psth[:,:,ix] = np.nanmean(T[:,:,idx_in],2)

#Split into N1 and N2 corresponding to V1 and all other visual areas 

    units['Row_number'] = np.arange(len(units))
    unitIDs1 = []
    unitIDs2 = []
    for index in units.index:
        structure_acronym = units.structure_acronym[index]
        if structure_acronym in areas1:
            unitIDs1.append(index)    
        if structure_acronym in areas2:
            unitIDs2.append(index)    

    unit_ind1 = units.Row_number[unitIDs1].values
    unit_ind2 = units.Row_number[unitIDs2].values

    N1 = len(unitIDs1)
    N2 = len(unitIDs2)

# DO THE TRANSFER ENTROPY


    print("performing transfer entropy calculation...")


    XX = Avg_psth

    te_mat = np.zeros([N1,N2,8])

    for itrial in range(8):

        print(itrial)
        for i in range(N1):
            for j in range(N2):

                symX = smite.symbolize(XX[unit_ind1[i],:,itrial],3)
                symY = smite.symbolize(XX[unit_ind2[j],:,itrial],3)

                TXY = smite.symbolic_transfer_entropy(symX, symY)
                TYX = smite.symbolic_transfer_entropy(symY, symX)
                
                te_mat[i,j,itrial]= TYX - TXY
    


    print("saving files...")

    save_object(te_mat, str(save_path +'/te_mat_' + stim_type + 'session_' + str(session_id) +'.pkl'))


# Average the orientations to get a units x units array


    session_TE = np.mean(te_mat, axis = 2)


    save_object(session_TE, str(save_path +'/session_TE_' + stim_type + 'session_' + str(session_id) +'.pkl'))

# Save the V1 units to df in depth order 

    VISp_units = units.loc[unitIDs1]

    save_object(VISp_units, str(save_path +'/VISp_units_' + stim_type + 'session_' + str(session_id) +'.pkl'))


# Save all visual units to df in depth order

    all_visual_units = units.loc[unitIDs2]

    save_object(all_visual_units, str(save_path +'/all_visual_units_' + stim_type + 'session_' + str(session_id) +'.pkl'))
    
    print("oh boy, my very own transfer entropy matrix")

