def xcorr_lag(session, rec_area, bin_width = 0.001, max_lag = 50):


    """ XCORR_LAG 
    
    Neuropixels data: function that given a session id, recording area (e.g. VISp) bin_width in ms (<10ms optimal) and maximum number of 
    lags on xcorr will return a matrix of correlation peak values between 0 and 1, and a matrix of lag bins as a function of your bin size
    and chosen max lags

    INPUTS:
        session_id     session from cache of neuropixels df
        rec_area       string of area desired (e.g. "VISp")
        bin_width      float in ms (<10ms optimal)
        max_lag        int - how many lags you want in your cross correlation 
    
    RETURNS: 
        peak_matrix    a num_units by num_units by num_unique_stimuli np array of xcorr score values between 0 and 1 
        lag_matrix     a num_units by num_units by num_unique_stimuli np array of the time lag with the peak xcorr score
        
    #e.g.

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.stats as stats
    import scipy.signal as sig
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    from allensdk.brain_observatory.ecephys import ecephys_session
    %matplotlib inline

    # fix slow autocomplete
    %config Completer.use_jedi = False

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
        data_root = "/run/media/tom.chartrand/Brain2019"

    manifest_path = os.path.join(data_root, "dynamic-brain-workshop/visual_coding_neuropixels/2019/manifest.json")

    cache = EcephysProjectCache.fixed(manifest=manifest_path)
    sessions = cache.get_sessions()

    corr_vals, lag_vals = xcorr_lag((cache.get_session_data(sessions.index.values[0]))), "VISp", 0.001, 50)

    #E.g. can plot with 

    fig,ax = plt.subplots(4, 2, figsize=(8,16))
    ax = ax.ravel()

    for i, a in enumerate(ax):
        a.imshow(peak_matrix[:,:,i], aspect="auto", cmap=plt.get_cmap('bwr'))
    

    ## numerous apologies for the inevitably numerous bugs ##
    """


    session_units = session.units
    
    rec_area_1 = rec_area
    rec_units_1 = session_units[session_units.structure_acronym == rec_area_1]
    rec_units_1 = rec_units_1.sort_values(by=['probe_vertical_position'], ascending=False)
    
    stim_table = session.get_presentations_for_stimulus(stimulus_type)
    stim_types = len(stim_table.unique())
    
    num_stims = len(stim_types)
    num_units = len(sample_units)
    
    peak_matrix = np.zeros((num_units, num_units, num_stims)) 
    lag_matrix = np.zeros((num_units, num_units, num_stims))
    
    bins = np.arange(0,0.5,bin_width)

    for trials in range(num_stims):    

        stim_presentation_ids = stimulus_ids

        histograms = session.presentationwise_spike_counts(
            bin_edges=bins,
            stimulus_presentation_ids=stim_presentation_ids,
            unit_ids=units.index.values
        )

        mean_histograms = histograms.mean(dim="stimulus_presentation_id")
        rates = mean_histograms/bin_width

        for i in range(num_units):

            for j in range(num_units):

                p = plt.xcorr(rates[:,i], rates[:,j], usevlines=True, maxlags=maxlags, normed=True, lw=2)

                corr_df = pd.DataFrame(p[1], p[0], columns = ["corr_R"])

                peak_corr = corr_df.sort_values(by = ["corr_R"], ascending = False)

                peak_lag = (peak_corr.corr_R[peak_corr.index.values[0]])

                if peak_corr.index[0] > 0:

                    peak_matrix[i, j, trials] = peak_lag

                elif peak_corr.index[0] == 0:
                    peak_matrix[i, j, trials] = np.inf

                elif peak_corr.index[0] < 0: 

                    peak_matrix[i, j, trials] = -(peak_lag)

                peak_lag = peak_corr.index.values[0]

                lag_matrix[i, j, trials] = peak_lag 

    return peak_matrix, lag_matrix