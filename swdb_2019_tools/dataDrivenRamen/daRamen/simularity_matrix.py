""" Calculate the Simularity Matrix """

# Model's Code
from Distance_calculations import likelihood_distance

# Python Standard Library
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


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

