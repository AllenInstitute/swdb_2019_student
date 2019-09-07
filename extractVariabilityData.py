#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:29:12 2019

@author: charles

Goal: process all mice, A4 + A6, B4 + B6 sessions, and save data from each

"""

#%% imports 
import os
import sys
import numpy as np
import pandas as pd

# matplotlib is a standard python visualization package
import matplotlib.pyplot as plt

# %matplotlib qt   # to put plots in new windows

# put support function folder on path:
sys.path.insert(1, '/home/charles/dynamicBrain2019/swdb_2019_student')
# import support fns:

from transformFiringRatesToLikelihoodMeasure import transformFiringRatesToLikelihoodMeasure
from temporalResponseVarianceHelperFunctions import findActiveCellsGivenStartTimes, addRewardedExcitationExperimentCheckoutTimesToExperimentTable, calculateAsymmetricDistributionStats 
# , makeHeatmap, plotCluesToEngagementVsNot, plotStatisticsOfVariableVariance, 

# seaborn is another library for statistical data visualization
# seaborn style & context settings make plots pretty & legible
import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white')
sns.set_palette('deep');

#%%
# Allen Brain import
# Import allensdk modules for loading and interacting with the data
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc

cache_path = '/media/charles/Brain2019/dynamic-brain-workshop/visual_behavior/2019'
cache = bpc.BehaviorProjectCache(cache_path)
 
#%%
''' Choose an experiment:  '''
# get the experiments as a panda:
experiments = cache.experiment_table

# select an experiment_id:
expList = experiments.loc[ ( experiments.passive_session== False ) & ( experiments.cre_line == 'Slc17a7-IRES2-Cre' ) ] 
# & (experiments.imaging_depth == 175) ] # 175 or 375.          inhibitory = 'Vip-IRES-Cre'

# Note: the 'index' col is not 0 to n, but rather are a subset of the indices of the full table
#%%
# add column with checkout times:
expList = addRewardedExcitationExperimentCheckoutTimesToExperimentTable(expList)
'''Special mouse: exp_id 864370674 checks out at 2700, but is also checked out 1000-2000. This is NOT added to the dataframe by this function.'''

exIdList = expList.ophys_experiment_id.values
indices = expList.index.values

spont = [1, 4, 9, 6]*60    # start and end of opening spont, start and end of closing spont (in seconds)
minSeenThreshold = 9     # minimum number of image flashes required, for each image, to allow use of a session.

# the images to process:
combinedImageList = ['im000', 'im106', 'im075', 'im073', 'im045', 'im054', 'im031', 'im035', 'im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im077','im085', 'omitted']
#%%
# Loop through the experiments. Return the number of each image in the post-checkout region, by image:   

numImsSeenCheckedOut = np.zeros([ len(exIdList), 9] )    # we know there are 9 ims ( 8 + 'omitted')
numImsSeenEngaged = np.zeros([ len(exIdList), 9 ])

expList['engagedImCounts'] = None
expList['checkedOutImCounts'] = None

for i in range(expList.shape[0]):
    # get the experiment_id
    ind = indices[i]
    exId = expList.ophys_experiment_id[ind]
    checkoutTime = expList.checkoutTime[ind]
    
    # make a titleStr:
    sessionNumber = expList.stage_name[ind]
    sessionNumber = sessionNumber[6]    # ie 1,2,3,4,5 or 6 as str
    depth = expList.imaging_depth[ind]
    titleStr = 'expID_' + str(exId) + '_Mouse_' + str(expList.animal_name[ind]) + '_' + expList.image_set[ind] + sessionNumber + '_' + str(depth) + 'um'

    # Given experiment_id, load this experiment (eg B4 for one mouse) from the cache.
    # sess will be an ExtendedBehaviorSession, Represents data from a single Visual Behavior Ophys imaging session. 
    sess = cache.get_session(exId)
    
    stimPres = sess.stimulus_presentations
    imList = np.unique(stimPres.image_name.values)
    
    numImsSeenEngaged = []
    numImsSeenCheckedOut = []
    for i2 in range(len(imList)):
        thisIm = imList[i2]
        startTimes = stimPres.loc[ (stimPres.image_name == thisIm) & (stimPres.change == False), 'start_time' ].values   # ignore change ims
        numImsSeenCheckedOut.append( np.sum(startTimes > checkoutTime) )
        numImsSeenEngaged.append( np.sum(startTimes < checkoutTime) )
        # special mouse:
        if exId == 864370674:
            temp = np.sum(np.logical_and(startTimes > 1000, startTimes < 2000))  # num ims seen in middle checked-out phase
            numImsSeenCheckedOut[i2] -= temp
            numImsSeenEngaged[i2] += temp
    expList.engagedImCounts[ind] = numImsSeenEngaged
    expList.checkedOutImCounts[ind] = numImsSeenCheckedOut
# heatmaps if wished:            
#makeHeatmap(numImsSeenEngaged, titleStr = 'ims seen engaged')
#makeHeatmap(numImsSeenCheckedOut, titleStr = 'ims seen checked out')
        
#
#for i in range  (10, 13):    #    (expList.shape[0]):
#    # get the experiment_id
#    ind = indices[i]
#    expList.engagedImCounts[ind] =  numImsSeenEngaged[i,:].reshape(1,-1) 
#    expList['checkedOutImCounts'][ind] =  numImsSeenCheckedOut[i,:].reshape(1,-1)

#%%
    
''' Collect data for each experiment pair B4, B6 (or A4, A6) where there are enough images. 
1. Condition to use: min # ims seen >= 9 in each session of pair
2. Collect both engaged and checked-out stats
2. calc active pixels on B4
3. On B6, calc stats using active pixels from B4

'''

# add many columns for each image to expList, to store results matrices:

# columns for statistics:
tag = 'engaged' 
for i in range(len(combinedImageList)):
    expList[ tag + 'ActiveCellIds_' + combinedImageList[i] ] = None
#for i in range(len(combinedImageList)):
#    expList['Delays_' + combinedImageList[i] ] =  None
#for i in range(len(combinedImageList)):
#    expList['Peaks_' + combinedImageList[i] ] =  None
     
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMean_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMeanStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMeanStdLeft_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedian_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedianStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedianStdLeft_' + combinedImageList[i] ] =  None   
   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMean_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMeanStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMeanStdLeft_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedian_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedianStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedianStdLeft_' + combinedImageList[i] ] =  None   
    

tag = 'checkedOut' 
for i in range(len(combinedImageList)):
    expList[ tag + 'ActiveCellIds_' + combinedImageList[i] ] = None
#for i in range(len(combinedImageList)):
#    expList['Delays_' + combinedImageList[i] ] =  None
#for i in range(len(combinedImageList)):
#    expList['Peaks_' + combinedImageList[i] ] =  None   
    
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMean_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMeanStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMeanStdLeft_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedian_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedianStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'LagMedianStdLeft_' + combinedImageList[i] ] =  None   
   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMean_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMeanStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMeanStdLeft_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedian_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedianStdRight_' + combinedImageList[i] ] =  None   
for i in range(len(combinedImageList)):
    expList[ tag + 'PeakMedianStdLeft_' + combinedImageList[i] ] =  None   

#%%
''' Loop through the experiments. We only process A1 and B4 (A1 includes also processing A3, B4 includes also processing B6)'''    
for i in range(expList.shape[0]):
    
    processEngagedCaseFlag = False
    processCheckedOutCaseFlag = False   
    ind = indices[i]
    
    if (expList.valid_cell_matching[ind]) & (expList.stage_name[ind] == 'OPHYS_4_images_B' ):      # ignore experiments without valid cell matching  # TEMPORARY: ALSO IGNORE 'A' SESSIONS
        exId = expList.ophys_experiment_id[ind]
        # identify if this is A1, B4, etc  
        sessionNumber = expList.stage_name[ind] 
        animal = expList.animal_name[ind]
        # imageSet = expList.image_set[ind]  # not needed
        matchingExId = -1    # dummy value
        if sessionNumber =='OPHYS_1_images_A':
            # find matching active session:
            matchingExId = int(expList.loc[ (expList.animal_name == animal) & (expList.stage_name == 'OPHYS_3_images_A') & (expList.valid_cell_matching == True), 'ophys_experiment_id'  ] )
        if sessionNumber == 'OPHYS_4_images_B':
            # find matching active session:
            matchingExId = int(expList.loc[ (expList.animal_name == animal) & (expList.stage_name == 'OPHYS_6_images_B') & (expList.valid_cell_matching == True), 'ophys_experiment_id'  ] )
        # see if there are enough checked out image views:
        # NOTE: This assumes expList has columns with counts of imagesSeen (inserted above)
        if  matchingExId > 0:    # ie there is a legit match
            minSeen1 = min(min(expList.loc[expList.ophys_experiment_id == exId, 'engagedImCounts' ].values))
            minSeen2 = min(min(expList.loc[expList.ophys_experiment_id == matchingExId, 'engagedImCounts' ].values)) 
            processEngagedCaseFlag = min([minSeen1, minSeen2])  >= minSeenThreshold
            minSeen1 = min(min(expList.loc[expList.ophys_experiment_id == exId, 'checkedOutImCounts' ].values)) 
            minSeen2 = min(min(expList.loc[expList.ophys_experiment_id == matchingExId, 'checkedOutImCounts' ].values)) 
            processCheckedOutCaseFlag = min([minSeen1, minSeen2]) >= minSeenThreshold
    # we now know whether to process the two cases for this pair
    
    if processEngagedCaseFlag or processCheckedOutCaseFlag:  # have to load session if either is True
        # load the first session
        sess = cache.get_session(exId)
        stimPres = sess.stimulus_presentations       
        
        dffTable = sess.dff_traces
        D = np.vstack(dffTable.dff.values)    # np array of one experiment's dff traces
        cellIds = dffTable.index.values
        T = sess.ophys_timestamps    # np vector of the experiment's timestamps
        # calc z-score matrix:
        startGrey =  np.logical_and(T > spont[0], T < spont[1] ) 
        endGrey = np.logical_and( T > max(T) - spont[2], T < max(T) - spont[3] )    
        W = startGrey + endGrey   # add the booleans
        zScoreDff, spontMean, spontStd = transformFiringRatesToLikelihoodMeasure( D, W )
        checkoutTime = expList.loc[ expList.ophys_experiment_id  == exId, 'checkoutTime' ].values 
        
        # get the subset of images to be processed that are in this session:
        imListForThisSession = np.unique(stimPres.image_name.values)
        imList = [ combinedImageList[i]  for i in   np.where( np.isin(combinedImageList, imListForThisSession) == True )[0]  ]
        
        # loop through images, getting stats for each:
        
        # engaged case:
        if processEngagedCaseFlag:
            tag = 'engaged'
            for i2 in range(len(imList)):
                imStarts = stimPres.loc[ (stimPres.image_name == imList[i2]) & (stimPres.change == False) ].start_time.values
                imStarts = imStarts[imStarts < checkoutTime ]
                # special mouse:
                if exId == 864370674:                          
                    imStarts = imStarts[np.logical_and( imStarts < 1000, startTimes > 2000) ]    # note direction of inequalities for engaged and for checked out (below)
                    
                # apply the function with default parameters:
                activeCellRowIndices, responseLags, ig2, dffPeaks, ig3, ig4 = findActiveCellsGivenStartTimes( D, imStarts, T, zScoreDff )                
                # get the cell_specimen_ids using the activeCellInds:
                activeCellIds = cellIds[activeCellRowIndices] 
                # set aside for use on second session:
                engagedActiveCellIds = activeCellIds 
                
                # store data: 
                thisRow = int(expList.loc[expList.ophys_experiment_id == exId].index.values) 
                
                # raw data:
                expList[ tag + 'ActiveCellIds_' + imList[i2]][thisRow] = activeCellIds 
#                expList[ tag + 'Delays_' + imList[i2] ][thisRow] =  responseLags[activeCellRowIndices, ]
#                expList[ tag + 'Peaks_' + imList[i2] ][thisRow ] =  dffPeaks[activeCellRowIndices, ]
                
                # calc and store lag variation statistics:
                Dactive = responseLags[activeCellRowIndices, ]
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'LagMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'LagMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'LagMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'LagMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'LagMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'LagMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft 
                
                # calc and store peak statistics vectors:
                Dactive = dffPeaks[activeCellRowIndices, ]
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'PeakMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'PeakMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'PeakMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'PeakMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'PeakMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'PeakMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft
                
                
                
        # checked out case is identical
        if processCheckedOutCaseFlag:
            tag = 'checkedOut'
            for i2 in range(len(imList)):
                imStarts = stimPres.loc[ (stimPres.image_name == imList[i2]) & (stimPres.change == False),'start_time' ].values
                imStarts = imStarts[imStarts > checkoutTime ]
                # special mouse:
                if exId == 864370674:                          
                    imStarts = imStarts[np.logical_and( imStarts > 1000, startTimes < 2000) ] 
                    
                # apply the function with default parameters:
                activeCellRowIndices, responseLags, ig2, dffPeaks, ig3, ig4 = findActiveCellsGivenStartTimes( D, imStarts, T, zScoreDff )                
                # get the cell_specimen_ids using the activeCellInds:
                activeCellIds = cellIds[activeCellRowIndices]  
                # set aside for use on second session:
                checkedOutActiveCellIds = activeCellIds
                thisRow = int(expList.loc[expList.ophys_experiment_id == exId].index.values) 
                expList[tag + 'ActiveCellIds_' + imList[i2]][thisRow] = activeCellIds 
#                expList[ tag + 'Delays_' + imList[i2] ][thisRow] =  responseLags[activeCellRowIndices, ]
#                expList[ tag + 'Peaks_' + imList[i2] ][thisRow ] =  dffPeaks[activeCellRowIndices, ]
                
                # calc and store lag variation statistics:
                Dactive = responseLags[activeCellRowIndices, ]
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'LagMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'LagMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'LagMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'LagMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'LagMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'LagMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft 
                
                # calc and store peak statistics vectors:
                Dactive = dffPeaks[activeCellRowIndices, ]
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'PeakMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'PeakMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'PeakMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'PeakMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'PeakMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'PeakMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft
                
        # Now repeat with the second session. The key difference is that we specify the active cells up front. We then restrict the dff matrix that goes into 'findActiveCells etc        
        ''' load the second session  '''
        sess = cache.get_session(matchingExId)
        stimPres = sess.stimulus_presentations       
        
        dffTable = sess.dff_traces
        D = np.vstack(dffTable.dff.values)    # np array of one experiment's dff traces
        
        T = sess.ophys_timestamps    # np vector of the experiment's timestamps
        # calc z-score matrix:
        startGrey =  np.logical_and(T > spont[0], T < spont[1] ) 
        endGrey = np.logical_and( T > max(T) - spont[2], T < max(T) - spont[3] )    
        W = startGrey + endGrey   # add the booleans
        zScoreDff, spontMean, spontStd = transformFiringRatesToLikelihoodMeasure( D, W )
        checkoutTime = expList.loc[ expList.ophys_experiment_id  == exId, 'checkoutTime' ].values  
        
        
        # get the subset of images to be processed that are in this session:
        imListForThisSession = np.unique(stimPres.image_name.values)
        imList = [ combinedImageList[i]  for i in   np.where( np.isin(combinedImageList, imListForThisSession) == True )[0]  ]
        
        # loop through images, getting stats for each:
        # engaged case:
        if processEngagedCaseFlag:
            tag = 'engaged'
            for i2 in range(len(imList)):
                imStarts = stimPres.loc[ (stimPres.image_name == imList[i2]) & (stimPres.change == False) ].start_time.values
                imStarts = imStarts[imStarts < checkoutTime ]
                # special mouse:
                if exId == 864370674:                          
                    imStarts = imStarts[np.logical_and( imStarts < 1000, startTimes > 2000) ]    # note direction of inequalities for engaged and for checked out (below)
                
                # restrict D to active cells from first session
                DActiveOnly  = D[ np.isin( dffTable.index.values, engagedActiveCellIds ), ]
                # apply the function with default parameters:
                ig1, responseLags, ig2, dffPeaks, ig3, ig4 = findActiveCellsGivenStartTimes( DActiveOnly, imStarts, T, zScoreDff )                
                
                thisRow = int(expList.loc[expList.ophys_experiment_id == matchingExId].index.values)   # row index in expList for storing 
                
                # store raw data
                expList[tag + 'ActiveCellIds_' + imList[i2]][thisRow] = engagedActiveCellIds 
#                expList[ tag + 'Delays_' + imList[i2] ][thisRow] =  responseLags
#                expList[ tag + 'Peaks_' + imList[i2] ][thisRow ] =  dffPeaks
                
                # calc and store lag variation statistics:
                Dactive = responseLags
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'LagMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'LagMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'LagMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'LagMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'LagMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'LagMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft 
                
                # calc and store peak statistics vectors:
                Dactive = dffPeaks
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'PeakMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'PeakMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'PeakMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'PeakMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'PeakMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'PeakMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft
                
                
        # checked out case is identical
        if processCheckedOutCaseFlag:
            tag = 'checkedOut'
            for i2 in range(len(imList)):
                imStarts = stimPres.loc[ (stimPres.image_name == imList[i2]) & (stimPres.change == False) ].start_time.values
                imStarts = imStarts[imStarts > checkoutTime ]
                # special mouse:
                if exId == 864370674:                          
                    imStarts = imStarts[np.logical_and( imStarts > 1000, startTimes < 2000) ] 
                    
                # restrict D to active cells from first session
                DActiveOnly  = D[ np.isin( dffTable.index.values, checkedOutActiveCellIds ), ]
                # apply the function with default parameters:
                ig1, responseLags, ig2, dffPeaks, ig3, ig4 = findActiveCellsGivenStartTimes( DActiveOnly, imStarts, T, zScoreDff )                 
                
                thisRow = int(expList.loc[expList.ophys_experiment_id == matchingExId].index.values) 
                expList[tag + 'ActiveCellIds_' + imList[i2]][thisRow] = checkedOutActiveCellIds 
#                expList[ tag + 'Delays_' + imList[i2] ][thisRow] =  responseLags
#                expList[ tag + 'Peaks_' + imList[i2]  ][thisRow] =  dffPeaks
                
                # calc and store lag variation statistics:
                Dactive = responseLags
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'LagMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'LagMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'LagMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'LagMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'LagMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'LagMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft 
                
                # calc and store peak statistics vectors:
                Dactive = dffPeaks
                meanVals, meanStdLeft, meanStdRight, medianVals, medianStdLeft, medianStdRight = calculateAsymmetricDistributionStats(Dactive)
                
                expList[ tag + 'PeakMean_' + imList[i2]][thisRow] = meanVals
                expList[ tag + 'PeakMeanStdRight_' + imList[i2]][thisRow] = meanStdRight
                expList[ tag + 'PeakMeanStdLeft_' + imList[i2]][thisRow] = meanStdLeft
                expList[ tag + 'PeakMedian_' + imList[i2]][thisRow] = medianVals
                expList[ tag + 'PeakMedianStdRight_' + imList[i2]][thisRow] = medianStdRight
                expList[ tag + 'PeakMedianStdLeft_' + imList[i2]][thisRow] = medianStdLeft
                
#%%                
''' Save this dataframe for future analysis'''
expList.to_csv('experimentTableWithCollectedData_full')















 
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    