import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
