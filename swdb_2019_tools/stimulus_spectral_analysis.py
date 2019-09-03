import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mtspec import mtspec
from mtspec.util import _load_mtdatas
from functools import reduce

def psth_multitaper_spectrum(session_id,unit_ids, stimulus_presentation_id, W,time_step = 1/100, 
                             temporal_frequency = None,orientation=None,phase=None,
                             spatial_frequency=None,show=True):
    '''
    PSTH_MULTITAPER_SPECTRUM computes the power spectrum for the peristimulus time histogram (PSTH) of an experiment 
    for a given mouse  using the multitaper method.  The PSTH is averaged over units and for each presentation of 
    the stimulus. The number of tapers is calculated using the recommended rule of thumb, 2 * T * W - 1, 
    where T is the duration of the stimulus presentation and W is the frequency resolution.  95% 
    confidence intervals are computed by using the jackknife method . 
    
    
    INPUTS:
        session_id                    = String, session_id of mouse
        unit_ids                      = Array, unit_ids
        stimulus_presentation_id      = String, simulus presentation name: e.g.'driftings_gratings' or 'spontaneous'..
        W                             = desired frequency resolution
        time_step                     = bin size for computing PSTH (default is 1/100)
        temporal_frequency (optional) = If not included, all possible temporal frequencies are included.
        orientation (optional)        = If not included, all possible orientations are included.
        phase (optional)              = If not included, all possible phases are included.
        spatial_frequency (optional)  = If not included, all possible spatial_frequencies are included.
        show                          = Boolean, signals whether to plot (default is True)
    
    RETURNS: 
        Pxx              = np array power up until Nyquist frequency
        f                = np array corresponding frequency axis
        jackknife        = np array corresponding error bars
        number_of_tapers = number of tapers used to computer the power spectrum
    
    '''
    # Get session and stimulus information
    session = cache.get_session_data(session_id)
    stim_table = session.get_presentations_for_stimulus(stimulus_presentation_id)
    stim_ids = get_stimulus_ids(stim_table,stimulus_presentation_id,orientation,temporal_frequency,phase,spatial_frequency)
    # Set power spectrum parameters
    duration = stim_table.duration.iloc[0]
    TW = int(duration * W)
    number_of_tapers = int(2 * TW - 1)
    fNQ = 1/time_step/2
    
    # Compute PSTH
    time_domain = np.arange(0,duration+time_step, time_step)

    histograms = session.presentationwise_spike_counts(bin_edges=time_domain, 
                                                   stimulus_presentation_ids=stim_ids, 
                                                   unit_ids=unit_ids)
    
    mean_histograms = histograms.mean(dim="stimulus_presentation_id")
    psth_trials = mean_histograms.values
    
    # Compute power spectrum
    Pxx, f, jackknife, _, _ = mtspec(data=psth_trials.mean(1),delta=time_step, time_bandwidth=TW,
            number_of_tapers=number_of_tapers, statistics=True)
    
    # Plot if true and return power spectrum & frequencies up until Nyquist frequency
    f_indices =f<=fNQ 
    f=f[f_indices]
    Pxx = Pxx[f_indices]
    jackknife = jackknife[f_indices,:]
    if show == True:
        fig,ax = plt.subplots(1,figsize=(15,4),sharex=True)
        plt.plot(f, Pxx.T,linewidth=3)
        ax.set_xlabel('Frequency (Hz)',fontsize=20)
        ax.set_ylabel('Power',fontsize=20)
        ax.legend(['Session ID '+ str(session_id)])
        ax.set_title('Power Spectrum',fontsize=20)
        ax.fill_between(f, jackknife[:, 0], jackknife[:, 1],
                 color="red", alpha=0.3)
        plt.xticks(np.arange(f[0], f[-1], step=1))
    return Pxx, f, jackknife, number_of_tapers

def get_stimulus_ids(stim_table,stimulus_presentation_id,orientation=None,temporal_frequency = None,phase=None,spatial_frequency=None):
    '''
    GET_STIMULUS_IDS outputs the stimulus IDs for input stimulus presentation ID and corresponding variables. Each
    stimulus presentation has a unique set of variables that change over the course of the experiments. 
    1. For 'gabors', orientation varies over experiment. If provided an orientation value, or is set to null, 
       function will return all stimulus presentations to that corresponding orientation.  If excluded, all orientations
       except for the null trials will be output.
    2. For 'drifting_gratings,' temporal frequency and orientation vary over experiment. If excluded, same as 1.
    3. For 'static_gratings,' spatial frequency, orientation and phase vary over experiment. If excluded, same as 1.
    4. For 'spontaneous','flashes','natural_movie_three','natural_movie_two', or 'natural scences', no variables
      change, so all are output. Flashes change color from -1 to 1, but this is not coded in here
      
    INPUTS:
      stim_table               = dataframe containing all stimulus presentations in an experiment for a stimulus.
      stimulus_presentation_id = String containing stimulus
      orientation              = (optional) Only for 'gabors', and 'drifting_gratings'.  If nothing is specified,
                                  then all orientations for given stimulus_presentation_id is output.
      temporal_frequency       = (optional) Only for 'drifting_gratings', and 'static_gratings.' If nothing 
                                  is specified, then all orientations for given stimulus_presentation_id is output.
      phase                    = (optional) Only for 'static_gratings' If nothing 
                                  is specified, then all orientations for given stimulus_presentation_id is output.
      spatial_frequency        = (optional) Only for 'static_gratings'If nothing 
                                  is specified, then all orientations for given stimulus_presentation_id is output.
      
    RETURNS:
      stim_ids                 = Array containing all indices corresponding to desired stimulus.
    
    '''
    ##### ---------SPONTANEOUS, FLASHES, NATURAL MOVIE ONE/THREE, NATURAL SCENES -----------
    if (stimulus_presentation_id == 'spontaneous') or (stimulus_presentation_id == 'flashes') or (stimulus_presentation_id == 'natural_movie_three') or (stimulus_presentation_id == 'natural_movie_two') or (stimulus_presentation_id == 'natural_scenes'):
        stim_ids = stim_table.index
    
    ##### ------------------GABOR FILTERS -------------------------------
    elif stimulus_presentation_id == 'gabors':
        if orientation is None:
            stim_ids = get_stim_ids_variable(stim_table.orientation,[0.0, 90.0, 45.0])
        else:
            stim_ids = get_stim_ids_variable(stim_table.orientation,[orientation])  
    ##### ------------------DRIFTING GRATINGS -------------------------------
    elif stimulus_presentation_id == 'drifting_gratings':
        if temporal_frequency is None:
            stim_ids1 = get_stim_ids_variable(stim_table.temporal_frequency,[1.0, 8.0, 15.0, 2.0, 4.0])
        else:
            stim_ids1 = get_stim_ids_variable(stim_table.temporal_frequency,[temporal_frequency])
            
        if orientation is None:
            stim_ids2 = get_stim_ids_variable(stim_table.orientation,[90.0, 0.0, 135.0, 315.0, 225.0, 180.0, 270.0, 45.0])
        else:
            stim_ids2 = get_stim_ids_variable(stim_table.orientation,[orientation])

        stim_ids = np.intersect1d(stim_ids1,stim_ids2)
        
    ##### ------------------STATIC GRATINGS -------------------------------        
    elif stimulus_presentation_id == 'static_gratings':
        
        if spatial_frequency is None:
            stim_ids1 = get_stim_ids_variable(stim_table.spatial_frequency,[0.32, 0.04, 0.16, 0.08, 0.02])
        else:
            stim_ids1 = get_stim_ids_variable(stim_table.spatial_frequency,[spatial_frequency])
            
        if orientation is None:
            stim_ids2 = get_stim_ids_variable(stim_table.orientation,[0.0, 90.0, 120.0, 60.0, 150.0, 30.0])

        else:
            stim_ids2 = get_stim_ids_variable(stim_table.orientation,[orientation])
            
        if phase is None:
            stim_ids3 = get_stim_ids_variable(stim_table.phase,[0.5, 0.0, 0.25, 0.75])
        else:
            stim_ids3 = get_stim_ids_variable(stim_table.phase,[phase])
        stim_ids = reduce(np.intersect1d, (stim_ids1, stim_ids2, stim_ids3))

    else:
        print('No stimulus input')
    return stim_ids

def get_stim_ids_variable(stim_table_key,values):
    '''
    GET_STIM_IDS_VARIABLE outputs all stim_ids containing the list of values.
    
    INPUTS:
    stim_table_key = Array containing all possible configurations for one variable and corresponding stim ID names. 
    values         = Array of values that could be in variable.
    
    OUTPUT:
    stim_ids       = Array of stim_ids that contain values.
    
    For example, if you want all orientations that are at 90 and 270 degrees:
       Run this line: get_stim_ids_variable(stim_table.orientation,[90,270])
    '''
    stim_ids = stim_table_key.isin(values).index

    return stim_ids