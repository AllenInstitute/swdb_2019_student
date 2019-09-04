#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Some helper functions for studying temporal variance of cell responses to start times (eg of images). '''

#%%

def findActiveCellsGivenStartTimes(D, starts, T, zScoreDff, windowLength = 0.5, activeMahalThresh = 2, minActiveFraction = 0.25):
    ''' 
    Finds the active cells for a given set of start times, subject to some parameters defining 'active'.
    
    CAUTION: This function does not contain error catches.
    
    Let n = total number of cells, s = total number of start times, t = total number of timepoints in experiment
    
    Parameters
    ----------
    D: np.array
        n x t. Each row is one unit's activity timecourse, each column corresponds to a timestamp.
        
    starts: np.array
        1 x t. The timestamps of the selected start times.

    T:  np vector
        1 x t. The timestamps for each column of D.
        
    zScoreDff: np.array
        n x t. It gives likelihood measures for each entry in D. It is the output of the function transformFiringRatesToLikelihoodMeasure( D, W )
    
    windowLength: scalar
        Time window in which to select peak response (seconds).
        
    activeMahalThresh: scalar.
        The minimum zScoreDff[i,j] can be for cell[i] to be considered active for startTime[j]. 'Mahal' refers to mahalanobis distance, a version of z-scores.
        
    minActiveFraction : scalar
        The fraction of start times a cell must be active on to count as active for that batch of start times.
        
    Returns
    -------
    activeCellInds: np column vector
        m x 1, where m = number of active cells. Indices of cells that were active.
        
    peakDelays: np.array
        n x s. Time lags between start time and cell peak value, for all cells and starts.
        
    activePercents: np vector
        n x 1. The percent of starts each cell was active for (all cells)
    
    dffPeaks: np.array
        n x s. The peak values for each cell and start.
        
    zScoreMax: np.array
        n x s. The peak z-scores for each cell and start.
        
    zScMaxZeroed: np.array
        n x s. zScoreMax, but with all values < 'activeMahalThresh' set to zero, to highlight active cells.
        
    Example
    -------
    (visual behavior dataset)
    # prepare inputs:
    D = np.vstack(session.dff_traces.values)    # np array of one experiment's dff traces 
    starts = imageTimes.loc[ (imageTimes.image_name == im) & (imageTimes.change == False) ].start_time.values
    T = session.ophys_timestamps.values    # np vector of the experiment's timestamps        
    W = np.logical_and(T > 30, T < 300)   # np boolean vector with 1s for timestamps > 30 seconds and < 5 minutes
    zScoreDff = transformToLikelihoodMeasure( D, W )        
    
    # apply the function with default parameters:
    activeCellInds, peakDelays, activePercents, dffPeaks, zScMax, zScMaxZeroed = findActiveCellsGivenStartTimes(D, starts, T, zScoreDff)        
    
    ---------    
    Sept 2 2019
    Charles Delahunt, delahunt@uw.edu 
    '''
    import numpy as np
    
    # initialize:
    activeInds = []     # each row will be a boolean of whether the response to that start time was above thresh
    zScMax = []
    dffPeaks = []
    peakDelays = []
    # go through starts:
    for t in starts: 
        window = np.logical_and( T >= t, T < t + windowLength).reshape(1,-1)
        window = np.where(window == 1)[1]
        miniT = T[ window ]
        zSc = np.max( zScoreDff[:, window ], axis = 1 )   # take max value
        # calculating delays takes a loop:
        dffTemp = np.max( D[ :,  window ], axis = 1 )  # col vector
        delayTemp = np.zeros(dffTemp.shape)
        for p in range(len(dffTemp)):
            delaySteps =  np.where( D[p, window] == dffTemp[p] )[0]  # the delay in timesteps
            delayTemp[ p ] = miniT[delaySteps] - t
            
        # append results for this start time:
        activeInds.append(zSc > activeMahalThresh)
        zScMax.append(zSc)
        dffPeaks.append(dffTemp)
        peakDelays.append(delayTemp)
     
    # make lists into arrays, each col corresponds to all cells and a fixed startTime:
    activeInds = np.vstack(activeInds)
    zScMax = np.vstack(zScMax)
    dffPeaks = np.vstack(dffPeaks)
    peakDelays = np.vstack(peakDelays)
    
    # some derived values:
    zScMaxZeroed =  np.multiply(zScMax, activeInds)
    activePercents = np.sum(activeInds, axis = 1) /100   # given the mahal thresh. all cells
 
    activeCellInds = np.where( activePercents > minActiveFraction)[0] 
    
    return activeCellInds, peakDelays, activePercents, dffPeaks, zScMax, zScMaxZeroed

#%%

def makeHeatmap(A, X =[], Y = [], yLabel = 'cells', xLabel = 'x', title = '', colorBarLabel = 'value', figSizeInches = (10,10) ):
    ''' 
    Plots a heatmap. This is code ripped from the visualBehavior tutorial and not optimized at all. X and Y ticks are not versatile, so X and Y are best left empty.
    
    Parameters
    ---------- 
        A: np.array
        X:values for x axis
        Y:values for y axis
        ylabel: str
            label for y axis
        xlabel: str
            label for x axis
        title: str
            label for title
        colorBarLabel: str
            label for colorbar
        figSizeInches: tuple
            eg (10,10)
    
    Returns
    -------
        Prints a heatmap
        
    '''
    
    from matplotlib import pyplot as plt
    import numpy as np
    
    if len(X) == 0:
        X = range(A.shape[1])
    if len(Y) == 0:
        Y = range(A.shape[0])
      
    fig, ax = plt.subplots(figsize=figSizeInches)
    cax = ax.pcolormesh(A, cmap='magma', vmin=0, vmax=np.percentile(A, 99))
    ax.set_yticks(np.arange(0, np.max(Y)), 10);
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)
    ax.set_title(title)
#ax.set_xticks(np.arange(0, len(sess.ophys_timestamps), 600*31));
    ax.set_xticklabels(np.arange(0, np.max(X), 600));
    plt.colorbar(cax, pad=0.015, label='dF/F')
    
