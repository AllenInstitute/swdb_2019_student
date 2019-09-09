import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys import ecephys_session

import sys
sys.path.append('../../swdb_2019_tools')
# import Neuropixels_timeseries
import spikeutils

import platform
platstring = platform.platform()

if 'Darwin' in platstring:
    # OS X 
    data_root = "/Volumes/Brain2019/"
elif 'Windows'  in platstring:
    # Windows (replace with the drive letter of USB drive)
    data_root = "E:/"
elif ('amzn1' in platstring):
    # then on AWS
    data_root = "/data/"
else:
    # then your own linux platform
    # EDIT location where you mounted hard drive
    data_root = "/media/$USERNAME/Brain2019/"

manifest_path = os.path.join(data_root, "dynamic-brain-workshop/visual_coding_neuropixels/2019/manifest.json")

cache = EcephysProjectCache.fixed(manifest=manifest_path)
sessions = cache.get_sessions()

#find all sessions with region of interest
hcareas = ['CA']
hc_sessions = []
for i in np.arange(len(sessions.structure_acronyms)):
    sessionid = sessions.structure_acronyms.index[i]
    if any(elem in sessions.structure_acronyms[sessionid] for elem in hcareas):
        hc_sessions.append(sessionid)


#define decoder dataframe to fill with for loop
decode_accuracy = pd.DataFrame(columns=['Session','Region','C'])


for i in range(len(hc_sessions)):

    seshid = hc_sessions[i]
#     seshid = 759883607
    #choose a session
    print('Loading session ' + str(seshid) + '...')

    #get all units and channels
    allunits = cache.get_units(annotate=True)
    channelinfo = cache.get_channels()

    #choose units that are on appropriate channels
    ca_channelinfo = channelinfo[channelinfo.manual_structure_acronym == hcareas[0]]
    ca_units = allunits[(allunits.peak_channel_id.isin(ca_channelinfo.index))&(allunits.ecephys_session_id == seshid)]


    #get session data for chosen session
    session = cache.get_session_data(seshid)


    #get spike times for session
    spike_times = session.spike_times

    #use following two lines to make a list of spiketimes
    ca_spiketimes = []
    for i in range(len(ca_units)):
        unit_id = ca_units.index[i]
        ca_spiketimes.append(spike_times[unit_id])

    driftgrat_presentations = session.get_presentations_for_stimulus(stimulus_names='drifting_gratings')
    natscene_presentations = session.get_presentations_for_stimulus(stimulus_names='natural_scenes')

    ori_45 = driftgrat_presentations[(driftgrat_presentations.orientation==45)&(driftgrat_presentations.temporal_frequency==4)]
    ori_315 = driftgrat_presentations[(driftgrat_presentations.orientation==315)&(driftgrat_presentations.temporal_frequency==4)]

#     nat_imga = natscene_presentations[natscene_presentations.frame==15]
#     nat_imgb = natscene_presentations[natscene_presentations.frame==72]

    ntrials_drift = len(ori_45)
#     ntrials_nat = len(nat_imga)

    ####FOR DRIFTING GRATINGS

    binsize = 1

    # #return 15 rows of binned fr
    # #stimrates_45 = np.zeros(15*binsize,numunits)  #trials by N
    # #stimrates_315 = np.zeros(15*binsize,numunits)

    numunits = len(ca_spiketimes)

    nbins_45 = np.int(np.ceil(((ori_45.stop_time.iloc[0]-(ori_45.start_time.iloc[0]))*1000)/binsize))
    nbins_315 = np.int(np.ceil(((ori_315.stop_time.iloc[0]-(ori_315.start_time.iloc[0]))*1000)/binsize))

    print(nbins_45)
    print(nbins_315)

    trialresponses_45 = np.zeros((nbins_45,numunits,ntrials_drift))
    for i in range(ntrials_drift):
        starttime = ori_45.start_time.iloc[i]
        endtime = ori_45.start_time.iloc[i]+2.0017
        trialresponses_45[:,:,i] = spikeutils.spiketimes_to_2D_rates(ca_spiketimes, starttime,
                                            endtime, binsize=binsize).T

    trialresponses_315 = np.zeros((nbins_315,numunits,ntrials_drift))
    for i in range(ntrials_drift):
        starttime = ori_315.start_time.iloc[i]
        endtime = ori_315.start_time.iloc[i]+2.0017
        trialresponses_315[:,:,i] = spikeutils.spiketimes_to_2D_rates(ca_spiketimes,starttime,endtime,binsize=binsize).T


    ####FOR NATURAL SCENES

#     nbins_nat1 = np.int(np.ceil(((nat_imga.stop_time.iloc[0]-(nat_imga.start_time.iloc[0]))*1000)/binsize))
#     nbins_nat2 = np.int(np.ceil(((nat_imgb.stop_time.iloc[0]-(nat_imgb.start_time.iloc[0]))*1000)/binsize))

#     trialresponses_nat1 = np.zeros((nbins_nat1,numunits,ntrials_nat))
#     for i in range(ntrials_nat):
#         starttime = nat_imga.start_time.iloc[i]
#         endtime = nat_imga.stop_time.iloc[i]
#         trialresponses_nat1[:,:,i] = spikeutils.spiketimes_to_2D_rates(ca_spiketimes, starttime,
#                                             endtime, binsize=binsize).T

#     trialresponses_nat2 = np.zeros((nbins_nat2,numunits,ntrials_nat))
#     for i in range(ntrials_nat):
#         starttime = nat_imgb.start_time.iloc[i]
#         endtime = nat_imgb.stop_time.iloc[i]
#         trialresponses_nat2[:,:,i] = spikeutils.spiketimes_to_2D_rates(ca_spiketimes, starttime,
#                                             endtime, binsize=binsize).T

    trialavgresponse_45 = np.mean(trialresponses_45,axis=2)
    trialavgresponse_315 = np.mean(trialresponses_315,axis=2)
#     trialavgresponse_nat1 = np.mean(trialresponses_nat1,axis=2)
#     trialavgresponse_nat2 = np.mean(trialresponses_nat2,axis=2)


    #FR in 200 bins after stimulus presentation
    stackedresponse_45 = trialresponses_45[0:200,:,:].sum(axis=0)
    stackedresponse_315 = trialresponses_315[0:200,:,:].sum(axis=0)
#     stackedresponse_nat1 = trialresponses_nat1[0:200,:,:].sum(axis=0)
#     stackedresponse_nat2 = trialresponses_nat2[0:200,:,:].sum(axis=0)

#     responses = np.vstack((stackedresponse_45.T,stackedresponse_315.T,stackedresponse_nat1.T,stackedresponse_nat2.T))
    responses = np.vstack((stackedresponse_45.T,stackedresponse_315.T))
    
    labels = np.zeros(responses.shape[0])
    block1end = ntrials_drift
    block2end = block1end + (ntrials_drift)
#     block3end = block2end + (ntrials_nat)
#     block4end = block3end + (ntrials_nat)

    labels[0:block1end] = 0
    labels[block1end:block2end] = 1
#     labels[block2end:block3end] = 2
#     labels[block3end:block4end] = 3

    stimuniques = list(np.unique(labels))
    stimuniques[0] = 'drifting gratings 45'
    stimuniques[1] = 'drifting gratings 315'
#     stimuniques[2] = 'natural scene 1'
#     stimuniques[3] = 'natural scene 2'


    plt.figure()
    plt.imshow(responses.T,aspect='auto');
    plt.xlabel("Time", fontsize=14);
    plt.ylabel("Cells", fontsize=14);
    # plt.vlines([block1end,block2end,block3end],ymin=0,ymax=numunits-1,color='white')
#     plt.vlines([ntrials_drift,(2*ntrials_drift),(2*ntrials_drift)+ntrials_nat],ymin=0,ymax=numunits-1,color='white')
#     plt.title('Drift 45, Drift 315, Nat Scene 15, Nat Scene 72')
    plt.vlines([ntrials_drift],ymin=0,ymax=numunits-1,color='white')
    plt.title('Drift 45, Drift 315')    
    plotname_fig1 = str(hcareas[0]) + 'drift_FR_session' + str(seshid) +'.png'
    filename_fig1 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig1)
    plt.savefig(filename_fig1,dpi=300)


    print('Performing PCA...')
    ####NOTE from PCA tutorial: probably better to normalize to the spontaneous region rather than to itself
    responses_norm = responses.copy()
    responses_norm -= responses.mean(axis=0)
    pca = PCA(n_components=15)
    pca.fit(responses_norm)

    plt.figure()
    plt.plot(pca.explained_variance_ratio_, '.')
    plt.xlabel("Component #", fontsize=14)
    plt.ylabel("Explained variance ratio", fontsize=14)
    plotname_fig2 = str(hcareas[0]) + 'drift_ScreePlot_session' + str(seshid) +'.png'
    filename_fig2 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig2)
    plt.savefig(filename_fig2,dpi=300)

    response_reduced = pca.fit_transform(responses_norm)

    plt.figure()
    plt.plot(response_reduced[:,0], response_reduced[:,1], '.')
    plt.xlabel("First principal component", fontsize=14)
    plt.ylabel("Second principal component", fontsize=14)
    plotname_fig3 = str(hcareas[0]) + 'drift_PCAunlabeled_session' + str(seshid) +'.png'
    filename_fig3 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig3)
    plt.savefig(filename_fig3,dpi=300)

    plt.figure()
    plt.plot(pca.components_[0])
    plt.xlabel("Cell #", fontsize=14)
    plt.ylabel("Loading_PC1", fontsize=14)
    plotname_fig4 = str(hcareas[0]) + 'drift_LoadingPC1_session' + str(seshid) +'.png'
    filename_fig4 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig4)
    plt.savefig(filename_fig4,dpi=300)

    plt.figure()
    plt.plot(pca.components_[1])
    plt.xlabel("Cell #", fontsize=14)
    plt.ylabel("Loading_PC2", fontsize=14)
    plotname_fig5 = str(hcareas[0]) + 'drift_LoadingPC2_session' + str(seshid) +'.png'
    filename_fig5 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig5)
    plt.savefig(filename_fig5,dpi=300)

    print(labels)
    plt.figure()
#     sns.scatterplot(x=response_reduced[:,0],y=response_reduced[:,1],hue=labels,palette='jet',legend='full')
    sns.scatterplot(x=response_reduced[:,0],y=response_reduced[:,1],hue=labels,legend='full')
    plt.xlabel("First principal component", fontsize=14)
    plt.ylabel("Second principal component", fontsize=14)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plotname_fig6 = str(hcareas[0]) + 'drift_PCA_withlabels_session' + str(seshid) +'.png'
    filename_fig6 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/'+ plotname_fig6)
    plt.savefig(filename_fig6,dpi=300)

    print('Performing LDA...')
    plt.xlabel("First principal component", fontsize=14)
    [X_train,X_test,y_train,y_test] = model_selection.train_test_split(response_reduced[:,0:2],labels,test_size=0.5)
    classifier = LDA()
    classifier.fit(X_train,y_train)
    y_hat = classifier.predict(X_test)
    plt.figure()
#     sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_hat,palette='jet',legend='full')
    sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_hat,legend='full')
    plt.ylabel("Second principal component", fontsize=14)
    plt.legend(stimuniques,bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plotname_fig7 = str(hcareas[0]) + 'drift_LDA_session' + str(seshid) +'.png'
    filename_fig7 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig7)
    plt.savefig(filename_fig7,dpi=300)

    print('Making confusion matrix, stashing results...')
    C = confusion_matrix(y_test,y_hat)
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    cax = ax.imshow(C,interpolation='none',origin='lower',vmin=0,vmax=C.max())
    ax.set_xlabel('Actual Class',fontsize=16)
    ax.set_ylabel('Predicted Class',fontsize=16)
#     ax.set_xticks(range(4))
    ax.set_xticks(range(2))
    ax.set_xticklabels(stimuniques,fontsize=16)
#     ax.set_yticks(range(4))
    ax.set_yticks(range(2))
    ax.set_yticklabels(stimuniques,fontsize=16)
    plt.colorbar(cax)
    plt.title(str('Session_'+ str(seshid) + '_Confusion'))
    plotname_fig8 = str(hcareas[0]) + 'drift_confusion_session' + str(seshid) +'.png'
    filename_fig8 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig8)
    plt.savefig(filename_fig8,dpi=300)

    df_sesh = {'Session':seshid,'Region':hcareas[0],'C':[C.ravel()]}
    print(df_sesh)
    decode_accuracy = decode_accuracy.append(df_sesh,ignore_index=True)


print(decode_accuracy)
print('Saving results...')
csvname_decode = str(hcareas[0]) + 'drift_LDA_accuracy' +'.csv'
filename_decode = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ csvname_decode) 
decode_accuracy.to_csv(filename_decode)

