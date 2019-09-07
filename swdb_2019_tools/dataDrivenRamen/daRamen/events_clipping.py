""" Modular Helper Functions for Clipping Event Related neural activity"""

import numpy as np


def get_event_related_1d(data, fs, orig_fs, indices, window, subtract_mean=None, overlapping=None, **kwargs):
    """Take an input time series, vector of event indices, and window sizes,
        and return a 2d matrix of windowed trials around the event indices.

        Parameters
        ----------
        data : array-like 1d
            Voltage time series
        fs : int
            Sampling Frequency
        orig_fs : int
            Original Sampling Frequency of Recorded Data
        indices : array-like 1d of integers
            Indices of event onset indices
        window : tuple | shape (start, end)
            Window (in ms) around event onsets, window components must be integer values
        subtract_mean : tuple, optional | shape (start, end)
            if present, subtract the mean value in the subtract_mean window for each
            trial from that trial's time series (this is a trial-by-trial baseline)
        overlapping : list, optional
            Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

        Returns
        -------
        event_related_matrix : array-like 2d
            Event-related times series around each index
            Each row is a separate event
        """

    def windows_to_indices(fs, window_times):
        """convert times (in ms) to indices of points along the array"""
        conversion_factor = (1 / fs) * 1000  # convert from time points to ms
        window_times = np.floor(np.asarray(window_times) / conversion_factor)  # convert
        window_times = window_times.astype(int)  # turn to ints

        return window_times

    def convert_index(fs, indexes, orig_fs):
        """convert the start times to their relative sample based on the fs parameter"""
        conversion_factor = (1 / fs) * orig_fs  # Convert from Original Sampling Rate to the Down sampled rate
        indexes = np.rint(np.array(indexes) / conversion_factor)
        indexes = indexes.astype(int)
        return indexes

    # Remove overlapping labels
    if overlapping is not None:
        overlaps = [index for index, value in enumerate(indices) if value in overlapping]  # Find overlapping events
        indices = np.delete(indices, overlaps, axis=0)  # Remove them from the indices

    window_idx = windows_to_indices(fs=fs, window_times=window)  # convert times (in ms) to indices
    inds = convert_index(fs=fs, indexes=indices,
                         orig_fs=orig_fs) + np.arange(window_idx[0], window_idx[1])[:, None]  # broadcast indices

    # Remove Edge Instances from the inds
    bad_label = []
    bad_label.extend([index for index, value in enumerate(inds[0, :]) if value < 0])  # inds that Start before Epoch
    bad_label.extend([index for index, value in enumerate(inds[-1, :]) if value >= len(data)])  # inds End after Epoch
    inds = np.delete(inds, bad_label, axis=1)  # Remove Edge Instances from the inds

    event_times = np.arange(window[0], window[1], (1 / fs) * 1000)
    event_related_matrix = data[inds]  # grab the data
    event_related_matrix = np.squeeze(event_related_matrix).T  # make sure it's in the right format

    # baseline, if requested
    if subtract_mean is not None:
        basewin = [0, 0]
        basewin[0] = np.argmin(np.abs(event_times - subtract_mean[0]))
        basewin[1] = np.argmin(np.abs(event_times - subtract_mean[1]))
        event_related_matrix = event_related_matrix - event_related_matrix[:, basewin[0]:basewin[1]].mean(axis=1,
                                                                                                          keepdims=True)

    return event_related_matrix


# TODO: Revisit the Function Below and its current Utility
def make_event_times_axis(window, fs):
    """
    Parameters
    ----------
    window : tuple (integers)
        Window (in ms) around event onsets
    fs : int
        Sampling Frequency

    Returns
    -------
    event_times : array
        Array of the Times Indicated in ms
    """
    event_times = np.arange(window[0], window[1], (1 / fs) * 1000)
    return event_times


def get_event_related_2d(data, indices, fs, orig_fs,  window, subtract_mean=None, overlapping=None, **kwargs):
    """
    Parameters
    ----------
    data: list, shape (Channels, Samples)
        Neural Data
    indices : list, shape [Events]
        Onsets of the Labels to be Clipped for one Chunk
    fs : int
            Sampling Frequency
    orig_fs : int
            Original Sampling Frequency of Recorded Data
    indices : array-like 1d of integers
            Indices of event onset indices
    window : tuple | shape (start, end)
        Window (in ms) around event onsets, window components must be integer values
    subtract_mean : tuple, optional | shape (start, end)
        if present, subtract the mean value in the subtract_mean window for each
        trial from that trial's time series (this is a trial-by-trial baseline)
    overlapping : list, optional
        Samples that overlap with another Epoch (overlapping_samples - epoch_abs_start)

    Returns
    -------
    events_matrix : ndarray | shape (Instances, Channels, Samples)
        Neural Data in the User Defined Window for all Instances of each Label
    """

    all_channel_events = np.apply_along_axis(func1d=get_event_related_1d, axis=-1, arr=data, fs=fs, orig_fs=orig_fs,
                                             indices=indices, window=window, subtract_mean=subtract_mean,
                                             overlapping=overlapping, **kwargs)
    events_matrix = np.transpose(all_channel_events, axes=[1, 0, 2])  # Reshape to (Events, Ch, Samples)
    return events_matrix
