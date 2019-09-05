#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:57:01 2019


Goal: decide 'checkout' times for B4, B6, A4, A6 for each mouse.


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
expList = experiments.loc[ ( experiments.passive_session== False ) & ( experiments.cre_line == 'Slc17a7-IRES2-Cre' ) ] 
# & (experiments.imaging_depth == 175) ] # 175 or 375.          inhibitory = 'Vip-IRES-Cre'

#%% 
exIdList = expList.ophys_experiment_id.values

for i in range(expList.shape[0]):
    ex_id = exIdList[i]
    sessionNumber = expList.stage_name.values[i]
    sessionNumber = sessionNumber[6]
    depth = expList.imaging_depth.values[i]
    titleStr = 'expID_' + str(ex_id) + '_Mouse_' + str(expList.animal_name.values[i]) + '_' + expList.image_set.values[i] + sessionNumber + '_' + str(depth) + 'um'

    # Given experiment_id, load this experiment (eg B4 for one mouse) from the cache.
    # sess will be an ExtendedBehaviorSession, Represents data from a single Visual Behavior Ophys imaging session. 
    sess = cache.get_session(ex_id)
    
    plotCluesToEngagementVsNot(sess, titleStr, saveFigFlag = True)