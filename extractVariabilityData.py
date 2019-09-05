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
from temporalResponseVarianceHelperFunctions import makeHeatmap, findActiveCellsGivenStartTimes, plotCluesToEngagementVsNot, plotStatisticsOfVariableVariance, addRewardedExcitationExperimentCheckoutTimesToExperimentTable 

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

#%%
# Loop through the experiments. Return the number of each image in the post-checkout region, by image:   
exIdList = expList.ophys_experiment_id.values
indices = expList.index.values
numImsSeenCheckedOut = np.zeros([ len(exIdList), 9] )    # we know there are 9 ims ( 8 + 'omitted')
numImsSeenEngaged = np.zeros([ len(exIdList), 9 ])

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
    
    for i2 in range(len(imList)):
        thisIm = imList[i2]
        startTimes = stimPres.loc[ (stimPres.image_name == thisIm) & (stimPres.change == False), 'start_time' ].values   # ignore change ims
        numImsSeenCheckedOut[i,i2] = np.sum(startTimes > checkoutTime)
        numImsSeenEngaged[i,i2] = np.sum(startTimes < checkoutTime)
        # special mouse:
        if exId == 864370674:
            temp = np.sum(np.logical_and(startTimes > 1000, startTimes < 2000))  # num ims seen in middle checked-out phase
            numImsSeenCheckedOut[i, i2] -= temp
            numImsSeenEngaged[i, i2] += temp
            
makeHeatmap(numImsSeenEngaged, titleStr = 'ims seen engaged')
makeHeatmap(numImsSeenCheckedOut, titleStr = 'ims seen checked out')
        
# add these values as new columns into the expList table:
# NOTE: assumed imList = 'im000', 'im031', 'im035', 'im045', 'im054', 'im073', 'im075', 'im106', 'omitted'
# CAUTION! These are entered as [[ 1,2,3,4, etc ]]  for unclear reasons (bad code)
expList['engagedImCounts'] = ''
expList['checkedOutImCounts'] = ''
for i in range(expList.shape[0]):
    # get the experiment_id
    ind = indices[i]
    expList.engagedImCounts[ind] =  numImsSeenEngaged[i,:].reshape(1,-1) 
    expList['checkedOutImCounts'][ind] =  numImsSeenCheckedOut[i,:].reshape(1,-1)

#%%
    
''' Collect data for each experiment pair B4, B6 (or A4, A6) where there are enough images. 
1. Condition to use: min # ims seen >= 9 in each session of pair
2. Collect both engaged and checked-out stats
2. calc active pixels on B4
3. On B6, calc stats using active pixels from B4




















 
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    