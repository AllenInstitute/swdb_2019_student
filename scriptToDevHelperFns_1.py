#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:53:21 2019

@author: charles

Goal: sandbox script to test fns that:
    1. select active cells given a list of times (eg image start times). Need defn of 'active' (z-score?)
    2. calc cell peak value latencies relative to stimulus onset (to calc temporal response variance)
    3. calc a cell's temporalresponse variance given a set of peak value latencies
    4. detect 'engaged' vs 'checked out' regions. Clues include running speed variance (high -> checked out); low lick rate; low reward rate; low TP to FP rate. 
    
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
from temporalResponseVarianceHelperFunctions import makeHeatmap, findActiveCellsGivenStartTimes, plotCluesToEngagementVsNot, plotStatisticsOfVariableVariance 

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
subsetOfExperiments = experiments.loc[ ( experiments.passive_session==False ) & ( experiments.cre_line == 'Slc17a7-IRES2-Cre' ) ] # or 'Vip-IRES-Cre' 
ex_id = 813083478
# subsetOfExperiments.ophys_experiment_id.sample(1).values[0]

# Given experiment_id, load this experiment (eg B4 for one mouse) from the cache.
# sess will be an ExtendedBehaviorSession, Represents data from a single Visual Behavior Ophys imaging session. 
sess = cache.get_session(ex_id)

# load relevant pandas:
dff = sess.dff_traces # dff traces panda: cell_specimen_id and dff
D = np.vstack(dff.dff.values)      # np array, each row is a cell's trace
T = sess.ophys_timestamps  # np vector, ophys times 
rewards = sess.rewards  # panda: timestamps, volume, autorewarded
lickTimes = sess.licks   # panda:  timestamps
# image info:
imageTimes = sess.stimulus_presentations   # panda: flash_id, image_index, start_time, omitted,  change, repeat_within_block, licks, rewards

# # hack to query on empty lists in 'licks'. Replace this with Michael's code from slack.
#a = imageTimes.licks.values
#licksNotEmpty = []
#for i in range(len(a)):
#    licksNotEmpty.append(len(a[i]))
#imageTimes['licksNotEmpty'] = licksNotEmpty

# running speed and dPrime:
run = sess.running_speed
roll = sess.get_rolling_performance_df()  # fn returns dataframe

#%%
# calc z-score matrix:
startGrey =  np.logical_and(T < 4.5*60, T > 0.67*60 ) 
endGrey = np.logical_and( T > max(T) - 9.6*60, T < max(T) - 5.6*60 )

W = startGrey + endGrey   # add the booleans 

zScoreDff, spontMean, spontStd = transformFiringRatesToLikelihoodMeasure( D, W )   # np array

#%%
''' fn 1: select active cells given a list of times '''

# here: im 0, non-change, yes licks. Ie FPs for im 0 (FP since these are not changes)
im = 'im000'
 
selectedStarts = imageTimes.loc[ (imageTimes.image_name == im) & (imageTimes.change == False) ] # & ( imageTimes.licksNotEmpty > 0) ] 
# Remove rows with empty list entries. These do not respond to dropna() or notnull():
removeEmptyEntryRowsMask = selectedStarts.licks.transform(len)!=0
selectedStarts = selectedStarts[ removeEmptyEntryRowsMask ] 
     
starts = selectedStarts.start_time.values


'''----------------------------------------------------------------------------------'''

#%% find the active cells:
activeCellInds, peakDelays, activePercents, dffPeaks, zScMax, zScMaxZeroed = findActiveCellsGivenStartTimes(D, starts, T, zScoreDff)

##%% make some plots:
#makeHeatmap(activeInds, X =[], Y = [], yLabel = 'cells', xLabel = 'starts', title = 'Active cells and starts matrix')
#makeHeatmap(zScMaxZeroed, X =[], Y = [], yLabel = 'cells', xLabel = 'starts', title = 'zScore Max')
#makeHeatmap(dffPeaks, X =[], Y = [], yLabel = 'cells', xLabel = 'starts', title = 'dffPeaks')   
#makeHeatmap(peakDelays[activeCellInds, :], X =[], Y = [], yLabel = 'cells', xLabel = 'starts', title = 'peak delays')
#
#fig = plt.figure(figsize=(5,5)) 
#plt.plot(np.sort(activePercents))
#plt.title('sorted fractions of start responses > activity mahal thresh, by cell') 

#%%

# call function to plot clues to engagement:
plotCluesToEngagementVsNot(sess, 'exp_id = ' + str(ex_id))
  

#%%
# visualize the histograms of active neurons only:
M = dffPeaks[activeCellInds[0:8], ]  # matrix where each row is a cell, responding to several flashes of an image
#M = peakDelays[activeCellInds, ]
binEdges = np.linspace(0, 0.5, 25)
plotStatisticsOfVariableVariance(M, binEdges, titleStr = 'distributions')

#%%
# calculate correlations between pairs of active neurons:
pda = peakDelays[activeCellInds, ]    # the peakDelay for active cells only

corrMatrix = np.corrcoef(pda, rowvar = True)

#### ERROR: manual correlation matrix fails (figure this out)
#pdaMean = np.mean(pda, axis = 1)
#pdaMeanZero = pda - np.tile(pdaMean.reshape(-1,1), [1, pda.shape[1] ])
#pdaStds = np.std(pda, axis = 1)
#denomMatrix = pdaStds.reshape(-1,1)*pdaStds.reshape(1,-1)
#numMatrix = np.matmul( pdaMeanZero, pdaMeanZero.T ) / pda.shape[1]  # divide by number of trials because numerator is a mean value of x*y if x and y are drawn from the mean-subtracted random variables X, Y
#corrMatrix = np.divide( numMatrix, denomMatrix )
 
makeHeatmap(corrMatrix, X =[], Y = [], yLabel = 'cells', xLabel = 'cells', title = 'correlations')












#%% 
''' calculate noise correlations for a group of cells, eg the active cells for a particular image ''' 
















