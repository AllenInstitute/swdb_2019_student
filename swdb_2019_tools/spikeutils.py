import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def spiketimes_to_firingrate(spiketimes,startime,stoptime,binsize,binarize=False): 
    '''
    Parameters
    ==========
    spiketimes : np.array
        array of spiketimes from one cell
    starttime, stoptime : float
        bounds of time to calculate firing rate between (in seconds)
    binsize : int
        size of bins (in milliseconds); use binsize = 1000 to get firing rate in Hz
    binarize : Boolean
        Binarize bins (returns array of 1's and 0's).  Default is False
    
    Returns
    =======
    counts : np.array
        array of counts within bins
    
    Example
    =======
    # Convert spiketimes of one cell to Hz (1000 ms bins) for a sample of 100 minutes
    spiketimes_to_firingrate(spiketrain,startime=0,stoptime=60*100,binsize=1000)
    --> array([1, 0, 0, ..., 3, 4, 1])
    '''
    startime_ms = startime * 1000
    stoptime_ms = stoptime * 1000
    binrange = np.arange(start = startime_ms,stop = stoptime_ms+1, step = binsize)

    spiketimes_cell = np.asarray(spiketimes) * 1000. #spiketimes in seconds * 1000 msec/sec
    counts, bins = np.histogram(spiketimes_cell,bins = binrange)
    if binarize == True:
        #binarize the counts
        counts[counts>0] = 1
    return(counts)

def spiketimes_to_2D_rates(spiketimes_list,startime,stoptime,binsize,binarize=False): 
    '''
    Parameters
    ==========
    spiketimes_list : list
        List of arrays of spiketimes, where each list element is the spiketimes array of a single cell
    starttime, stoptime : float
        bounds of time to calculate firing rate between (in seconds)
    binsize : int
        size of bins (in milliseconds); use binsize = 1000 to get firing rate in Hz
    binarize : Boolean
        Binarize bins (returns array of 1's and 0's).  Default is False
    
    Returns
    =======
    spikerates_2D : np.array
        N x T array of binned counts where rows are cells and columns are binned time
    
    Example
    =======
    # Convert spiketimes of multiple cell to Hz (1000 ms bins) for a sample of 100 minutes
    spiketimes_list
    --> [array([6.62989190e-01, 3.89575906e+00, 4.99826015e+00, ...,
        9.27489710e+03, 9.27502220e+03, 9.27511023e+03]),
     array([6.28122489e-01, 6.54955849e-01, 7.28389255e-01, ...,
        9.27778883e+03, 9.27783327e+03, 9.27785537e+03]),
     array([5.62389091e-01, 6.27822489e-01, 6.68522529e-01, ...,
        9.27784967e+03, 9.27787497e+03, 9.27789327e+03])]
    
    spiketimes_to_2D_rates(spiketimes_list,startime=0,stoptime=60*100,binsize=1000)
    --> array([[ 1,  0,  0, ...,  3,  4,  1],
           [17, 18, 33, ..., 25, 26, 38],
           [ 6, 18, 15, ..., 28, 29, 28]])
    '''
    startime_ms = startime * 1000
    stoptime_ms = stoptime * 1000
    binrange = np.arange(start = startime_ms,stop = stoptime_ms+1, step = binsize)
    n_cells = len(spiketimes_list)

    spikewords_array = np.zeros([n_cells,binrange.shape[0]-1])
    for i in range(n_cells):
        spiketimes_cell = np.asarray(spiketimes_list)[i] * 1000. #spiketimes in seconds * 1000 msec/sec
        counts, bins = np.histogram(spiketimes_cell,bins = binrange)
        if binarize == True:
            #binarize the counts
            counts[counts>0] = 1
        spikewords_array[i,:] = counts
    spikerates_2D = spikewords_array
    return(spikerates_2D)

def fs_rsu_split(units_to_split,cutoff,return_hist=False):
    ''' Accepts a pandas dataframe of units and splits units into fast-spiking and regular-spiking.
        Useful for cell type analysis of neuropixels data.
    Parameters
    ==========
    units_to_split : pandas.DataFrame
        As returned by cache.get_units method
    cutoff : float
        Value of waveform duration to split cells on
    return_hist : Boolean
        If true, returns the histogram of waveform duration for units_to_split.
        Useful for validating accurate cutoff.
    
    Returns
    ==========
    rsu_units : pandas.DataFrame
        DataFrame of Regular Spiking Units (putative pyramidal cells and some interneurons)
    fs_units : pandas.DataFrame
        DataFrame of Fast Spiking Units (putative PV interneurons)
    
    Example
    =======
    >>> rsu_cells, fs_cells = fs_rsu_split(some_units,0.4,return_hist=True)
    '''
    
    if ('duration' in units_to_split.columns) == True:
        if return_hist == True:
            plt.hist(units_to_split.duration,bins=60)
            plt.xlim(0,2)
            plt.xlabel('Waveform Duration (ms)')
            plt.ylabel('Number of units')

        rsu_units = units_to_split[units_to_split.duration>cutoff]
        fs_units = units_to_split[units_to_split.duration<cutoff]

    elif ('waveform_duration' in units_to_split.columns) == True:
        if return_hist == True:
            plt.hist(units_to_split.waveform_duration,bins=60)
            plt.xlim(0,2)
            plt.xlabel('Waveform Duration (ms)')
            plt.ylabel('Number of units')

        rsu_units = units_to_split[units_to_split.waveform_duration>cutoff]
        fs_units = units_to_split[units_to_split.waveform_duration<cutoff]
    
    return rsu_units, fs_units