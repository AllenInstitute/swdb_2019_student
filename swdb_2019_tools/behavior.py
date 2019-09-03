# Module for getting behavior features.
import numpy as np

def get_start_lick(licks, windows_size):
    """
    Find the start timestamp of a lick. 
    licks: numpy array
    window_size: second. 
    Return a numpy array of start timestamp. 
    """
    new_licks = []
    for idx in range(1, len(licks)):
        if licks[idx-1] < licks[idx] - window_size:
            new_licks.append(licks[idx])
    return np.asarray(new_licks)


def get_end_lick(licks, windows_size):
    """
    Find the end timestamp of a lick. 
    licks: numpy array
    window_size: second. 
    Return a numpy array of start timestamp. 
    """
    new_licks = []
    for idx in range(len(licks)-1):
        if licks[idx] + window_size < licks[idx+1]:
            new_licks.append(licks[idx])
    return np.asarray(new_licks)