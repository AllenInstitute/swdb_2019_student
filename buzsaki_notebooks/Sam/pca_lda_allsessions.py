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

# fix slow autocomplete
#%config Completer.use_jedi = False

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


############################## ACQUIRE DATA ##############################
cache = EcephysProjectCache.fixed(manifest=manifest_path)
sessions = cache.get_sessions()

#find all sessions with region of interest
hcareas = ['VISp']
hc_sessions = []
for i in np.arange(len(sessions.structure_acronyms)):
    sessionid = sessions.structure_acronyms.index[i]
    if any(elem in sessions.structure_acronyms[sessionid] for elem in hcareas):
        hc_sessions.append(sessionid)


#define decoder dataframe to fill with for loop
decode_accuracy = pd.DataFrame(columns=['Session','Region','spont_true','spont_false',
                                        'driftgrat_false','driftgrat_true'])

for i in range(len(hc_sessions)):
# for i in range(1):
    ############################## Session units and spiketimes #############################
    #choose a session
    seshid = hc_sessions[i]
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

    ############################## Stimulus Presentations ##############################
    stim_epochs = session.get_stimulus_epochs()
    stim_epochs.columns
    numepochs = len(stim_epochs.start_time)

    #Find unique stimulus epochs
    epochlabels, epochuniques = pd.factorize(stim_epochs.stimulus_name)

    ############################## Bin spikes ##############################
    print('Binning spikes')
    ## Make NxT array with array of times for each N
    #Generate binned spiketimes matrix
    binsize = 1000
    # endtime = stim_epochs.stop_time.values[-1]
    starttime=0
    endtime = np.ceil(np.max([np.max(spikes) for spikes in ca_spiketimes]))
    epochrates = spikeutils.spiketimes_to_2D_rates(ca_spiketimes,starttime,endtime,binsize=binsize).T

    #assign labels
    numlabels = len(np.unique(stim_epochs.stimulus_name))
    np.arange(numlabels)
    labels = np.zeros((int(endtime*1000/binsize)))
    for i in range(numepochs):
        epochstart = (stim_epochs.start_time[i]*1000)/binsize
        epochend = (stim_epochs.stop_time[i]*1000)/binsize
        stimname = stim_epochs.stimulus_name[i]
        labels[int(epochstart):int(epochend)] = np.where(epochuniques==stimname)[0][0]


    ############################## PCA ##############################
    print('Performing PCA...')
#     plt.figure()
#     plt.imshow(epochrates,aspect='auto');
#     plt.xlabel("Time", fontsize=14);
#     plt.ylabel("Cells", fontsize=14);
#     plotname_fig1 = str(hcareas[0]) + '_FR_session' + str(seshid) +'.png'
#     filename_fig1 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig1)
#     plt.savefig(filename_fig1,dpi=300)

    ####NOTE from PCA tutorial: probably better to normalize to the spontaneous region rather than to itself
    epochrates_norm = epochrates.copy()
    epochrates_norm -= epochrates.mean()

    pca = PCA(n_components=10)
    pca.fit(epochrates_norm)

#     plt.figure()
#     plt.plot(pca.explained_variance_ratio_, '.')
#     plt.xlabel("Component #", fontsize=14)
#     plt.ylabel("Explained variance ratio", fontsize=14)
#     plotname_fig2 = str(hcareas[0]) + '_ScreePlot_session' + str(seshid) +'.png'
#     filename_fig2 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig2)
#     plt.savefig(filename_fig2,dpi=300)

    response_reduced = pca.fit_transform(epochrates_norm)

#     plt.figure()
#     plt.plot(response_reduced[:,0], response_reduced[:,1], '.')
#     plt.xlabel("First principal component", fontsize=14)
#     plt.ylabel("Second principal component", fontsize=14)
#     plotname_fig3 = str(hcareas[0]) + '_PCAunlabeled_session' + str(seshid) +'.png'
#     filename_fig3 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig3)
#     plt.savefig(filename_fig3,dpi=300)

#     plt.figure()
#     plt.plot(pca.components_[0])
#     plt.xlabel("Cell #", fontsize=14)
#     plt.ylabel("Loading_PC1", fontsize=14)
#     plotname_fig4 = str(hcareas[0]) + '_LoadingPC1_session' + str(seshid) +'.png'
#     filename_fig4 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig4)
#     plt.savefig(filename_fig4,dpi=300)

#     plt.figure()
#     plt.plot(pca.components_[1])
#     plt.xlabel("Cell #", fontsize=14)
#     plt.ylabel("Loading_PC2", fontsize=14)
#     plotname_fig5 = str(hcareas[0]) + '_LoadingPC2_session' + str(seshid) +'.png'
#     filename_fig5 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig5)
#     plt.savefig(filename_fig5,dpi=300)

    #weird change in shape from epochrates to response_reduced, so I deal with it here
    # by ensuring labels and response_reduced are same shape
    response_reduced = response_reduced[0:labels.shape[0],:]

#     plt.figure()
#     sns.scatterplot(x=response_reduced[:,0],y=response_reduced[:,1],hue=labels,palette='jet',legend='full')
#     plt.xlabel("First principal component", fontsize=14)
#     plt.ylabel("Second principal component", fontsize=14)
#     plt.legend(epochuniques,bbox_to_anchor=(1, 1), loc='upper left')
#     plt.tight_layout()
#     plotname_fig6 = str(hcareas[0]) + '_PCA_withlabels_session' + str(seshid) +'.png'
#     filename_fig6 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig6)
#     plt.savefig(filename_fig6,dpi=300)

    label_natscene = np.where(epochuniques=='spontaneous')[0][0]
    label_driftgrat = np.where(epochuniques=='drifting_gratings')[0][0]

    proj_natscene = response_reduced[labels==label_natscene]
    proj_driftgrat = response_reduced[labels==label_driftgrat]

    lab_natscene = labels[labels==label_natscene]
    lab_driftgrat = labels[labels==label_driftgrat]

    lab_NatDrift = np.append(lab_natscene,lab_driftgrat,axis=0)
    proj_NatDrift = np.vstack((proj_natscene,proj_driftgrat))

    epochuniq_NatDrift = epochuniques[np.unique(np.array(lab_NatDrift,dtype=int))]

    plt.figure()
    sns.scatterplot(x=proj_NatDrift[:,0],y=proj_NatDrift[:,1],hue=lab_NatDrift,palette='jet',legend='full')
    plt.xlabel("First principal component", fontsize=14)
    plt.ylabel("Second principal component", fontsize=14)
    plt.legend(epochuniq_NatDrift,bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plotname_fig7 = str(hcareas[0]) + '_PCA_spontdrift_session' + str(seshid) +'.png'
    filename_fig7 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig7)
    plt.savefig(filename_fig7,dpi=300)
    print('Performing LDA...')
    # [X_train,X_test,y_train,y_test] = model_selection.train_test_split(response_reduced[:,0:2],labels,test_size=0.05)
    [X_train,X_test,y_train,y_test] = model_selection.train_test_split(proj_NatDrift[:,0:2],lab_NatDrift,test_size=0.05)

    classifier = LDA()
    classifier.fit(X_train,y_train)
    y_hat = classifier.predict(X_test)

    plt.figure()
    sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_hat,palette='jet',legend='full')
    plt.xlabel("First principal component", fontsize=14)
    plt.ylabel("Second principal component", fontsize=14)
    plt.legend(epochuniq_NatDrift,bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plotname_fig8 = str(hcareas[0]) + '_LDA_spontdrift_session' + str(seshid) +'.png'
    filename_fig8 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig8)
    plt.savefig(filename_fig8,dpi=300)
    print('Making confusion matrix, stashing results...')
    C = confusion_matrix(y_test,y_hat)
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111)
    cax = ax.imshow(C,interpolation='none',origin='lower',vmin=0,vmax=C.max())
    ax.set_xlabel('Actual Class',fontsize=16)
    ax.set_ylabel('Predicted Class',fontsize=16)
    ax.set_xticks(range(2))
    ax.set_xticklabels(epochuniq_NatDrift,fontsize=16)
    ax.set_yticks(range(2))
    ax.set_yticklabels(epochuniq_NatDrift,fontsize=16)
    plt.colorbar(cax)
    plt.title(str('Session_'+ str(seshid) + '_Confusion'))
    plotname_fig9 = str(hcareas[0]) + '_confusion_spontdrift_session' + str(seshid) +'.png'
    filename_fig9 = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ plotname_fig9)
    plt.savefig(filename_fig9,dpi=300)

    df_sesh = {'Session':seshid,'Region':hcareas[0],'spont_true':C[0,0],'spont_false':C[0,1],
                                            'driftgrat_false':C[1,0],'driftgrat_true':C[1,1]}
    print(df_sesh)
    decode_accuracy = decode_accuracy.append(df_sesh,ignore_index=True)

print(decode_accuracy)
print('Saving results...')
csvname_decode = str(hcareas[0]) + '_LDA_accuracy_spontdrift' +'.csv'
filename_decode = os.path.abspath(os.getcwd()+'/../../buzsaki_plots/PCA_LDA/'+ csvname_decode) 
decode_accuracy.to_csv(filename_decode)
