#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def transformFiringRatesToLikelihoodMeasure(A, W):
    """
        Description
        -----------
        Transforms a firing rate (FR) matrix into a matrix that reflects the likelihood (z-scores) of the FRs relative to a window of spontaneous activity. Each unit's FR timecourse is normalized by the standard deviation of that unit's FR activity in the specified window. The new timecourse gives the mahalanobis distances (z-scores) from spontaneous activity of that timestamp's activity. 
        The z-scores give a different perspective on unit activity: Apparently active units may be inactive from a z-score viewpoint because their responses are usually inside their spontaneous noise envelope, while apparently quiet units may be highly active from a z-score viewpoint because their responses are often outside their spontaneous noise envelope.
        Assumes symmetric std deviations, and uses mean (as opposed to median) as the center of the distribution.
        
        CAUTION: This function does not contain error catches.
        
        Parameters
        ----------
        A: np.array
            n x t. Each row is one unit's activity timecourse, each column corresponds to a timestamp.
    
        W:  np.array of booleans
            boolean array that gives the indices of timestamps inside the spontaneous activity window.      
        
        Returns
        -------
        M: np.array
            n x t. Each row is one unit's activity timecourse measured as the mahalanobis distance from its spontaneous activity in the window W.
            
        spontMean: np.array
            n x 1 vector. The i'th entry is the mean of the i'th unit's responses in the spontaneous noise window.
            
        spontStd: np.array
            n x 1 vector. The i'th entry is the std of the i'th unit's responses in the spontaneous noise window.
            
        Example
        -------
        (visual behavior dataset)
        A = np.vstack(session.dff_traces.values)    # np array of one experiment's dff traces 
        T = session.ophys_timestamps.values    # np vector of the experiment's timestamps
        W = np.logical_and(T > 30, T < 300)   # np boolean vector with 1s for timestamps > 30 seconds and < 5 minutes
        M, spontMean, spontStd = transformToLikelihoodMeasure( A, W )
        
        Sept 2 2019
        Charles Delahunt, delahunt@uw.edu
"""    
    import numpy as np
    
    spont = A[:, W == 1]            # FRs in spontaneous region

    spontStd = np.std(spont, axis = 1).reshape(-1,1)
    spontMean = np.mean(spont, axis = 1).reshape(-1,1)
    
    # subtract the spontMean, then divide by spontStd to get z-score: 
    M = np.divide( A - np.tile(spontMean, [ 1, A.shape[1] ] ),
                  np.tile(spontStd,[ 1, A.shape[1] ]) )  
    
    return M, spontMean, spontStd
    
