#!/usr/bin/env python
# coding: utf-8

# In[167]:


from matplotlib.patches import Patch

def plot_dff_stim_overlay(ts, dff, epochs, num_traces):
    """
    Purpose: create the dff plots with overlay, expanded figure from tutorial
    
    Requires: matplotlib
    
    Inputs:
    ts: numpy array of time stamps
    
    dff: numpy array of the dff trace magnitudes
    
    epochs:pandas DataFrame of the stimulus epochs, including stimulus, start, and end data columns
    
    num_traces: int number of traces to plot
    
    Returns: 
    Matplotlib figure
    
    Example: 
    plot_dff_stim_overlay(ts,dff1,epochs, num_traces)
    """
    fig = plt.figure(figsize=(10*2,8))
    for i in range(num_traces):
        fig = plt.plot(ts/60, dff[i,:]+(i*2), color='gray')

    colors = ['blue', 'orange', 'green', 'red', 'cyan', 'yellow', 'white']
    sample_to_ts = (ts[-1]-ts[0])/len(ts)
    stim_names = epochs.stimulus.unique();
    #legend_elements = []
    legend_elements = [Patch(color=colors[i[0]], alpha=0.2, label=i[1]) for i in enumerate(stim_names)]
    for c, stim_name in enumerate(stim_names):
        stim1 = epochs[epochs.stimulus == stim_name]
        for j in range(len(stim1)):
            fig = plt.axvspan(xmin = stim1.start.iloc[j]*sample_to_ts/60, xmax=stim1.end.iloc[j]*sample_to_ts/60, color = colors[c], alpha=0.1)
        fig = plt.xlabel('Time in Minutes')
        fig = plt.ylabel('DFF Traces of {} Neurons'.format(num_plots))
        fig = plt.title('DFF Traces Across Stim Conditions')
        fig = plt.legend(handles = legend_elements, loc=2, borderaxespad=0.)
    return fig


# In[157]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[158]:


dff1=np.load('./shared/DFF_C2_275.npy')
ts = np.load('/efs/Courtnie2894/ts.npy')
epochs= pd.read_pickle('/efs/Courtnie2894/stim_epoch_start.pkl')
num_traces=30


# In[166]:


plot_dff_stim_overlay(ts,dff1,epochs, num_traces)


# In[ ]:




