def sanity_check(mat,xname='x-axis(unknown units)',yname = 'y-axis(unknown units)', zname = 'z-axis(unknown units)',title = 'Unknown matrix',ncols=5):
    """given a numpy 3D array the function prints its size and a color plot of each layer along the third dimansion.
    This function could be used for plotting spike counts across trials or whatever you like
    
    Example: inspect a matrix containing neural spike counts for 6 units along 50 time bins and 6 trials
    
    #create a random test activity matrix
    import numpy as np
    activity_matrix = np.rand.random(6,50,6)
    sanity_check(activity_matrix)
    
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    N = mat.shape
    
    print(['the size of your matrix is' + str(N)])
    nrows =(N[2]//ncols)
    
    # plot each trial in a subplot
    fig,axes = plt.subplots(nrows,ncols, figsize=(18,nrows*4), sharex=True, sharey=True)
    axes = axes.ravel()
    for itrial,ax in enumerate(axes):
        ploti = ax.imshow(mat[:,:,itrial],aspect='auto')
        plt.colorbar(ploti,ax=ax)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(zname + ' ' + str(itrial))
    
    plt.suptitle(title)
    return 