def psth_multitaper_spectrum(session_id,unit_ids, stimulus_presentation_id, W,time_step = 1/100, 
                             orientation=None,temporal_frequency = None,phase=None,
                             spatial_frequency=None,show=True):
    '''
    PSTH_MULTITAPER_SPECTRUM computes the power spectrum for the peristimulus time histogram (PSTH) of an experiment 
    for a given mouse  using the multitaper method.  The PSTH is averaged over units and for each presentation of 
    the stimulus. The number of tapers is calculated using the recommended rule of thumb, 2 * T * W - 1, 
    where T is the duration of the stimulus presentation and W is the frequenct resolution.  Error bars are 
    computed by XXX. 
    
    INPUTS:
        session_id                    = String, session_id of mouse
        unit_ids                      = pandas.core.indexes.numeric.Int64Index, unit_ids
        stimulus_presentation_id      = String, 'driftings_gratings' or 'xxx'..
        temporal_frequency (optional) = ??
        orientation (optional)        = ??
        W                             = desired frequency resolution
        time_step                     = bin size for computing PSTH (default is 1/100)
        show                          = Boolean, signals whether to plot (default is True)
    
    RETURNS: 
        Pxx              = np array power up until Nyquist frequency
        f                = np array corresponding frequency axis
        jackknife        = np array corresponding error bars
        number_of_tapers = number of tapers used to computer the power spectrum
    
    '''
    session = cache.get_session_data(session_id)
    stim_table = session.get_presentations_for_stimulus(stimulus_presentation_id)
#    if stimulus_presentation_id == 'spontaneous' || stimulus_presentation_id == 'flashes' ||
#        stimulus_presentation_id == 'natural_movie_three' || stimulus_presentation_id == 'natural_movie_two'||
#        stimulus_presentation_id == 'natural_scenes':
#        print('Stim presented ID is', stimulus_presentation_id )
#    elif stimulus_presentation_id == 'gabors':
#        print()
#    elif stimulus_presentation_id == 'drifting_gratings':
#    elif stimulus_presentation_id == 'static_gratings':
#    else:
#        print('No stimulus input')
        
    stim_ids = stim_table[(stim_table.orientation==orientation)&
                          (stim_table.temporal_frequency==temporal_frequency)].index
    
    duration = stim_table.duration.iloc[0]
    TW = int(duration * W)
    number_of_tapers = int(2 * TW - 1)
    
    
    time_domain = np.arange(0,duration+time_step, time_step)
    
    histograms = session.presentationwise_spike_counts(bin_edges=time_domain, 
                                                   stimulus_presentation_ids=stim_ids, 
                                                   unit_ids=v1_units.index)
    
    mean_histograms = histograms.mean(dim="stimulus_presentation_id")
    
    psth_trials = mean_histograms.values
    Pxx, f, jackknife, _, _ = mtspec(data=psth_trials.mean(1),delta=time_step, time_bandwidth=TW,
            number_of_tapers=number_of_tapers, statistics=True)
    fNQ = 1/time_step/2

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
