import math
import time
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Train a network to match a set of target functions
def trainRNN(targets, g, dt, tau, noiseLevel=0, P0alpha=1, trainRate=1, minDeltaError=0.00001, maxRuns=10000):
    '''
    Parameters
    ==========
    targets : 2-dimensional numpy array
        array of time series where rows designate units and columns designate time
    g : float
        this number dictates the amount of chaos with which the random network is initialized. A g value of less than 1 or smaller will intialize a network with no chaotic properties. Recommended values range from 1.3 to 2.
    dt : float
        number denoting the timestep length. This should be the sampling frequency of the data in Hz.
    tau : float
        number restricting the amount of change in current per timestep for each unit. This value is recommended to be ~10x dt.
    noiseLevel : float
        if you would like to input noise into your network during training, insert a value for the strength of the noise. A typical noiseLevel value would be 0.1. Default is 0.
    P0alpha : float or int
        a number used for initialization which controls the learning rate. Typically, numbers from 1 to 10 are effective. Default is 1.
    trainRate : int
        a number denoting which timesteps training will occur on. A value of 5 would mean updating of the J matrix occurs on every 5th timestep. Default is 1.
    minDeltaError : float
        deltaError is calculated after each run (mean absolute error of the run) and the deltaErrors of the last four runs are used to compare to the minDeltaError. Learning will continue until the deltaError is less than the minDeltaError. Default is 0.00001.
    maxRuns : int
        the maximum number of runs the network should train for before stopping. Default is 10000.

    Returns
    =======
    J : 2-dimensional numpy array
        Returns a numpy array of N x N size (N designating the number of units in the target). This is the trained Wij weight matrix. This weight matrix can be used to run the network across time to analyze the network's dynamical properties. It can also be used to assess properties of connectivity of the network.
    errors : list of floats
        Returns a list of mean absolute errors for each run across training. This can be plotted to provide an idea of the convergence of the network across training.

    Example
    =======
    # Trains a network to match imported time series data
    targets = genfromtxt('./hypothetical_data.csv', delimiter=',')
    Wij, errs = trainRNN(targets, 1.5, 1/30)
    '''
    N = targets.shape[0]
    T = np.arange(0,targets.shape[1]*dt,dt)
    sigN = noiseLevel * math.sqrt(tau / dt)

    J = g * rd.randn(N, N) / math.sqrt(N)
    
    errors = []
    deltaError = minDeltaError + 1
    run_error = N + 1
    runs = 0
    PJ = P0alpha * np.eye(N,N)
    while (deltaError > minDeltaError or run_error > N/100) and runs < maxRuns: 
        runs = runs + 1
        Rates = np.zeros([N,len(T)])
        current = targets[:,0]
        Rates[:,0] = current
        run_error = 0
        for t in range(len(T)):
            Rates[:,t] = np.tanh(current)
            JR = (J @ Rates[:,t]) #+ sigN * np.random.randn(N,)
            current = (-current + JR)*dt/tau + current
            if t % trainRate == 0:
                err = JR - targets[:,t] # e(t) = z(t) - f(t)
                run_error = run_error + np.mean(err ** 2)
                Pr =  PJ @ Rates[:,t]
                rPr = Rates[:,t] @ Pr
                c = 1.0 / (1.0 + rPr)
                PJ = PJ - np.outer(Pr,Pr)*c
                J = J - (c * np.outer(err,Pr))
        if len(errors) < 5:
            deltaError = minDeltaError + 1
        else:
            errordiffs = []
            for e in range(4):
                errordiffs = np.append(errordiffs,
                                       (errors[len(errors)-e-2] - errors[len(errors)-e-1]))
            deltaError = np.mean(np.abs(errordiffs))
        errors = np.append(errors, run_error)
    return J, errors

# Run network across time using a given Wij matrix (J), and compare time series activity to targets
def networkRates(targets, J, dt, tau, plot=False, numUnits=10):
    '''
    Parameters
    ==========
    targets : 2-dimensional numpy array
        array of time series where rows designate units and columns designate time
    J : 2-dimensional numpy array
        array of N x N size (N designating the number of units in the target). This is the trained Wij weight matrix, the output from the trainRNN() function.
    dt : float
        number denoting the timestep length. This should be the sampling frequency of the data in Hz.
    tau : float
        number restricting the amount of change in current per timestep for each unit. This value is recommended to be ~10x dt.
    plot : boolean
        whether or not you would like a plot of the network activity overlaid on the target time series. Default is False.
    numUnits : int
        number of units you would like plotted. This is only used if the plot argument is True. Default is 10.
    '''
    N = J.shape[0]
    T = np.arange(0, targets.shape[1]*dt, dt)
    networkRates = np.zeros([N, len(T)])
    current = targets[:,0]
    for t in np.arange(0,len(T)):
        networkRates[:,t] = np.tanh(current) # Add rate to traces
        JR = np.matmul(J,networkRates[:,t])
        current = (-current + JR)*dt/tau + current # Update current
    if plot:
        trainedUnitsPlot = plt.figure(figsize=(20,10))
        for i in range(numUnits):
            plt.plot(np.arctanh(networkRates[i,:]) + i, linewidth=1.2, color="salmon")
            plt.plot(targets[i,:] + i, linewidth=1.5, linestyle=":", color="darkblue")
            plt.ylabel("Rate")
            plt.title("Red is Trained Network; Blue is Target")
        network_patch = mpatches.Patch(color='salmon', label='Network')
        targets_patch = mpatches.Patch(color='darkblue', label='Target')
        plt.legend(handles=[network_patch, targets_patch], loc='upper right')
        plt.xlabel("Time (ms)")
        plt.show()
        return networkRates
    else:
        return networkRates
            