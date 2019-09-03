""" Calculate the Simularity Matrix """

# Model's Code
import LLSA_calculations as lvarc

# Python Standard Library
import numpy as np
import numpy.ma as ma
from multiprocessing import Pool, cpu_count
from functools import partial


def compute_master_theta(models, windows_w, tseries_w):
    master_tseries = []
    for model in models:
        worm, window_idx = model
        window = windows_w[window_idx]
        t0, tf = window
        ts = tseries_w[worm][t0:tf]
        master_tseries.append(ts)
        master_tseries.append([np.nan] * ts.shape[1])
    master_tseries = ma.masked_invalid(ma.vstack(master_tseries))
    master_theta, eps = lvarc.get_theta_masked(master_tseries)
    return master_theta


def likelihood_distance(models, windows_w, tseries_w, thetas_w):
    master_theta = compute_master_theta(models, windows_w, tseries_w)
    distances = []
    for model in models:
        worm_idx, model_idx = model
        theta = thetas_w[model_idx]
        window = windows_w[model_idx]
        t0, tf = window
        ts = ma.masked_invalid(tseries_w[worm_idx][t0:tf])
        theta_here, eps = lvarc.get_theta(ts)
        distances.append(lvarc.loglik_mvn_masked(theta, ts) - lvarc.loglik_mvn_masked(master_theta, ts))
    return np.sum(distances)


def bad_scope(models, all_windows, all_tseries, all_thetas, indexes):
    index_x, index_y = indexes
    sel_models = models[[index_x, index_y]]
    distance = likelihood_distance(sel_models, all_windows, all_tseries, all_thetas)  # Calculate Distances
    return index_x, index_y, distance


def compute_dissimilarity_matrix(models, all_windows, all_tseries, all_thetas, n_jobs=1):
    """ Compute the Dissimilarity Matrix

    Paramters
    ---------
    models : ndarray
        Index of the trained linear models
    all_windows : ndarray
        Index of the windows corresponding to the linear models
    all_tseries : list | shape (models, parameters, inputs)
        List of the time series used to train models
    all_thetas : ndarray | shape = (trials, samples, inputs) #TODO: Change the dimensions to have samples last-dim
        List of the parameters corresponding to the trained linear models
    n_jobs : int, optional, default: 1
        Number of jobs to run in parallel.
        1 is no parallelization. -1 uses all but 1 available cores.

    Returns
    -------
    dissimilarity_matrix : ndarray | shape = (models, models)
        Strict Lower Triangle of the Dissimilarity matrix
    """
    num_models = len(models)  # Number of Models
    dissimilarity_matrix = np.zeros((num_models, num_models))  # Initialize the array
    index_i, index_j = np.tril_indices(n=num_models, k=-1)  # Get the Indexes of the strict lower triangle

    if n_jobs == 1:
        for index_x, index_y in zip(index_i, index_j):
            sel_models = models[[index_x, index_y]]
            distance = likelihood_distance(sel_models, all_windows, all_tseries, all_thetas)  # Calculate Distances
            dissimilarity_matrix[index_x, index_y] = distance  # Populate the Matrix
    else:

        n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
        n_jobs = min(n_jobs, num_models)
        print("Getting Distances (parallel, n_jobs={0}/{1} total)".format(n_jobs, cpu_count()))

        pool = Pool(processes=n_jobs)
        fixed_bad_scope = partial(bad_scope, models, all_windows,
                                  all_tseries, all_thetas)

        for index_x, index_y, distance in pool.imap_unordered(fixed_bad_scope, zip(index_i, index_j)):
            dissimilarity_matrix[index_x, index_y] = distance  # Populate the Matrix

    return dissimilarity_matrix
