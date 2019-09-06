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
        m x 1, where m = number of active cells. Row indices of cells that were active. 
        !!!! CAUTION !!!! These are NOT 'cell_specimen_id' values
        
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
    dffTable = session.dff_traces
    D = np.vstack(dffTable.dff_traces.values)    # np array of one experiment's dff traces 
    starts = imageTimes.loc[ (imageTimes.image_name == im) & (imageTimes.change == False) ].start_time.values
    T = session.ophys_timestamps.values    # np vector of the experiment's timestamps        
    W = np.logical_and(T > 30, T < 300)   # np boolean vector with 1s for timestamps > 30 seconds and < 5 minutes
    zScoreDff = transformToLikelihoodMeasure( D, W )        
    
    # apply the function with default parameters:
    activeCellInds, peakDelays, activePercents, dffPeaks, zScMax, zScMaxZeroed = findActiveCellsGivenStartTimes(D, starts, T, zScoreDff)
    
    # get the cell_specimen_ids using the activeCellInds:
    cellIds = dffTable.index.values
    activeCellSpecimenIds = cellIds[activeCellInds]       
    
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
    if len(activeInds) > 0:
        activeInds = np.vstack(activeInds)
    if len(zScMax) >0:
        zScMax = np.vstack(zScMax)
    if len(dffPeaks) > 0:
        dffPeaks = np.vstack(dffPeaks)
    if len(dffPeaks) > 0:
        peakDelays = np.vstack(peakDelays)
    
    # some derived values:
    if min(len(zScMax), len(activeInds) ) > 0:
        zScMaxZeroed =  np.multiply(zScMax, activeInds)
    else:
        zScMaxZeroed = zScMax
    if len(activeInds) > 0:
        activePercents = np.sum(activeInds, axis = 1) /100   # given the mahal thresh. all cells
        activeCellInds = np.where( activePercents > minActiveFraction)[0] 
    else:
        activePercents = []
        activeCellInds = [] 
    
    return activeCellInds, peakDelays, activePercents, dffPeaks, zScMax, zScMaxZeroed

#%%
def plotCluesToEngagementVsNot(sess, titleStr = '', saveFigFlag = False):
    ''' Plot various clues as to whether the mouse is engaged or checked out. Uses lick rate, reward rate, rolling d_prime, and std of windowed running speed. 
    There are many hard-coded parameters, eg windowSize
    
        Parameters
        ---------- 
            sess: BehaviorOphysSession class instance
                From a single experiment_id. This is eg B4 for a single mouse
                
            titleStr: str
                What you want in the title of the figures
                
            saveFigFlag: boolean
        
        Returns
        -------
            Plot with four subplots:
                
            1. Lick rate vs time
            
            2. (smoothed) Reward rate vs time
            
            3. Std dev of windowed running speed vs time
            
            4. Rolling D_prime vs time
            
       Example
       -------
       plotCluesToEngagementVsNot(sess, titleStr = 'experiment_id = 123' )
       
       Charles Delahunt, delahunt@uw.edu. 4 sept 2019
    ''' 
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    T = sess.ophys_timestamps  # np vector, ophys times 
    rewards = sess.rewards  # panda: timestamps, volume, autorewarded
    lickTimes = sess.licks   # panda:  timestamps 
    roll = sess.get_rolling_performance_df()  # fn returns dataframe
    run = sess.running_speed
    # remove outliers from run data:
    runTemp = run.speed.values
    clipValHigh = np.percentile(runTemp, 99)
    clipValLow = np.percentile(runTemp, 1)
    run.speed = np.maximum( np.minimum(runTemp, clipValHigh), clipValLow )
    
    
    # varused for both licks and rewards:
    Tsubsampled = T[0:-1:50]  # roughly one per second
    
    # plt.subplots(figsize = (10,20))
    fig, axes = plt.subplots(nrows=4, ncols=1,figsize = (10,20), sharex = True)

#df1.plot(ax=axes[0,0])
#df2.plot(ax=axes[0,1])
    
    ####### 1.  get lick rates vs time and plot:
    windowSize = 120    # seconds.  
    lickVector = lickTimes.timestamps.values
    lickRate = np.zeros(Tsubsampled.shape)
    for i in range(len(Tsubsampled)):
        lickRate[i] = np.sum(abs(lickVector - Tsubsampled[i]) < windowSize/2) / windowSize *60  # licks per minute
         
    #plt.figure(figsize = (10,5))
    plt.subplot(4,1,1)
    plt.plot(Tsubsampled, lickRate)
    #plt.xlabel('time (s)')
    plt.ylabel('licks/min')
    plt.title( titleStr ) # 'lick rate')
    plt.grid(b=True)
    plt.xlim([0,4500])
    
    ####### 2. Get rewards rates vs time and plot:
    windowSize = 180    # seconds.  
    rewardsVector = rewards.timestamps.values
    rewardsRate = np.zeros(Tsubsampled.shape)
    for i in range(len(Tsubsampled)):
        rewardsRate[i] = np.sum( abs( rewardsVector - Tsubsampled[i] ) < windowSize/2 ) / windowSize*60  # licks per minute
    
    # smooth:
    ham = np.hamming(20)
    rewardsRateSmoothed = np.convolve(rewardsRate, ham, mode='same')    
    #plt.figure(figsize = (10,5))
    plt.subplot(4,1,2)
    plt.plot(Tsubsampled, rewardsRateSmoothed)
    #plt.xlabel('time (s)')
    plt.ylabel('rewards/min')
    #plt.title('reward rate')
    plt.grid(b=True)
    plt.xlim([0,4500])
    
    ## raster plot:
    #plt.figure(figsize = (20,5))
    #plt.plot(rewardsVector, np.ones(rewardsVector.shape),'*')
    #plt.title('rewards raster')
    #plt.xlabel('time (s)')
    
    ####### 3.  plot variation in run speed:
    
    # window run:  # timestamp step = 0.02 sec, so 3000 = 1 minute
    windowSize = 12000
    runWindowed = run.rolling(windowSize)
    run['runSpeedMedian'] = runWindowed.median().speed
    run['runSpeedStd'] = runWindowed.std().speed
    plt.subplot(4,1,3)
    #ax = plt.gca()
    run.plot(x = 'timestamps', y = [ 'runSpeedMedian', 'runSpeedStd' ], ax = plt.gca() ) #,  title =  titleStr) # figsize = (10,5), title =  titleStr)
    plt.grid(b=True)
    plt.xlim([0,4500])
    plt.ylabel('running')
    plt.xlabel('')
    
    ####### 4.  rolling d prime:
    # remove nans:
    roll['rolling_dprime'][roll.rolling_dprime.isnull()] = 0
    # get timestamps for each dprime:
    roll['timesForDprimes'] = sess.trials.change_time.values
    # plot rolling dPrime:
    plt.subplot(4,1,4) 
    roll.plot( x = 'timesForDprimes', y = 'rolling_dprime',  ax = plt.gca() ) #, title =  titleStr) # figsize = (10,5), title = titleStr) 
    plt.grid(b=True)
    plt.xlim([0,4500])
    plt.ylabel('dPrime')
    plt.xlabel('time (s)')

    if saveFigFlag:
        figName = titleStr + '.png'
        fig.savefig(figName)
        plt.close(fig)
        
    # run.plot(x = 'timestamps', y = 'speed')     # instantaneous run speed, useful to check effect of clipping
    
#%%
def plotStatisticsOfVariableVariance(M, binEdges = 50, titleStr = ''):
    ''' 
    Given a matrix where each row is a cell's responses to different trials, (a) plot the histograms of each cell, (b) plot the mean vs the median (as a measure of non-gaussian-ness).
    
    Parameters
    -----------
        M : np.array
            Array of neural responses, eg time lags or peak values, where each row is a neuron and each column is a trial
            
        binEdges : list OR scalar
            Used as hist argin 'bin'.
            
        titleStr : str
            To add to titles.
        
    Returns
    --------
        Two plots:
            1. histograms of neuron response distributions, in gridded subplots
            
            2. scatterplot of median vs mean, as a check on skewness
            
    Example
    --------
        M = peakVal  # matrix where each row is a cell, responding to several flashes of an image
        binEdges = np.linspace(0, 0.5, 50)
        plotStatisticsOfVariableVariance(M, binEdges, titleStr = 'distributions')
            
    Charles Delahunt, delahunt@uw.edu. 4 sept 2019
    '''
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    # 1. plot histograms in grid:
    if len(binEdges) == 1:
        binEdges = np.linspace(0,max(M.flatten()),50)
        
    plt.figure(figsize = (15, 15)) 
    numCols = 5
    numRows =  M.shape[0] // numCols + 1    # will give an extra row if 5 divides exactly
    for i in range( M.shape[0] ):
        plt.subplot( numRows, numCols, i + 1) 
        plt.hist( M[i, ], bins = binEdges )   # leave out times, just use indices
        plt.title( 'cell (row) index ' + str(i) ) 
        
    # calculate mean and median:
    Mmean = np.mean(M, axis = 1)
    Mmedian = np.median(M, axis = 1)
    # delayDistStd = np.std(peakDelays, axis = 1)   # not used
    
    # plot mean vs median to get sense of skew:
    plt.figure()
    plt.plot([min(Mmean), max(Mmean)],[min(Mmean), max(Mmean)], 'g')
    plt.plot(Mmean, Mmedian, 'k*') 
    plt.title(titleStr + ' median vs mean')
    plt.xlabel('mean')
    plt.ylabel('median')
    plt.grid(b = True)
    plt.axis('equal')
#%%

def makeHeatmap(A, X =[], Y = [], yLabel = 'cells', xLabel = 'x', titleStr = '', colorBarLabel = 'value', figSizeInches = (10,10) ):
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
    ax.set_title(titleStr)
#ax.set_xticks(np.arange(0, len(sess.ophys_timestamps), 600*31));
    ax.set_xticklabels(np.arange(0, np.max(X), 600));
    plt.colorbar(cax, pad=0.015, label='dF/F')
    
#%% 
 
def addRewardedExcitationExperimentCheckoutTimesToExperimentTable(expTable):
    ''' Given a subset of the experiment_table that excludes inhibitory neuron experiments and passive experiments: add a column with the manually-collected checkout times (ie when the mouse disengaged) in seconds.  Requires no repeats in the last 4 digits of experiment_id values.
This must be done each time because the order of the table can change each time it is summoned. Note that a few 4-digit expIds have an invisible leading zero
Special mouse: exp_id 0674 checks out at 2700, but is also checked out 1000-2000. This is NOT added to the dataframe by this function.

    Parameters
    ---------
        expTable: pandas dataframe
    
    Returns: 
        expTable: pandas dataframe
        
    Example:
        expList = experiments.loc[ ( experiments.passive_session== False ) & ( experiments.cre_line == 'Slc17a7-IRES2-Cre' ) ] 
        expList = addRewardedExcitationExperimentCheckoutTimesToExperimentTable(expList)
    '''
    import numpy as np
    # The manually-collected checkout times:
    expIdAlignedToCheckoutTimes= np.array([6687,2719,3334,9543,1034,1118,3478,7785,7518,7135,141,781,9926,9496,8936,3730,5577,2970,4639,7604,7625,9305,9605,6106,467,1524,8115,2951,2969,7488,6766,7033,8066,674,2085,3088,1157,1028,3858,1992,6128,5304,5766])
    checkoutTimes = np.array([4000,4000,4000,3000,2600,3800,2500,2000,2500,2100,4000,3000,3600,2500,2400,4000,3400,4000,3200,3000,3300,3200,4000,4000,3400,3800,2400,2500,4000,1500,4000,4000,1500,2700,3500,3700,3600,4000,4000,4000,4000,4000,3400])
    checkoutOrderedByCurrentExpList = np.zeros(checkoutTimes.shape)
    expIdList = expTable.ophys_experiment_id.values
    for i in range(len(expIdList)):
        this = expIdList[i] % 10000
        index = np.where( expIdAlignedToCheckoutTimes == this )[0]
        if len(index) > 0:    # catch against other experiments being in table
            checkoutOrderedByCurrentExpList[i] = checkoutTimes[ index  ]
        else:
            checkoutOrderedByCurrentExpList[i] = -1
    expTable['checkoutTime'] = checkoutOrderedByCurrentExpList 
    
    return expTable
   