def sanity_check(mat,xname='x-axis(unknown units)',yname = 'y-axis(unknown units)', zname = 'z-axis(unknown units)',title = 'Unknown matrix',ncols=5):
    """given a numpy 3D array the function prints its size and a colorplot of each layer along the third dimension.
    This function could be used to visually inspect 3D matrices (e.g. plotting spike counts across trials and units)
    
    INPUTS mat: 2D or 3D numpy array that you would like to inspect
           x-axis (optional): string containing the label for first array's dimension
           y-axis (optional): string containing the label for second array's dimension
           z-axis (optional): string containing the label for third array's dimension
           title(optional): string containing the sup title for the plot
           ncols(optional): int number of columns to use for the subplots (default = 5)

    OUTPUTS plot the matrix using the 
    Example: inspect a matrix containing neural spike counts for 6 units along 50 time bins and 6 trials
    
    #create a random test activity matrix
    import numpy as np
    activity_matrix = np.rand.random(6,50,6)
    sanity_check(activity_matrix)
    
    # if you are working in jupiter notebook and want to visualize your plot inline add the following line before calling the   function: 
    %matplotlib inline
    
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    N = mat.shape
    print(['the size of your matrix is' + str(N)])
    
    if len(N)==2:
        plt.imshow(mat,aspect='auto')
        plt.colorbar()
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(zname)
    elif N[2]<100:
        if N[2]%ncols == 0:
            nrows =(N[2]//ncols)
        else:
            nrows = (N[2]//ncols)+1
        # plot each trial in a subplot
        fig,axes = plt.subplots(nrows,ncols, figsize=(18,nrows*4), sharex=True, sharey=True)
        axes = axes.ravel()
        axes = axes[range(N[2])]
        for itrial,ax in enumerate(axes):
            ploti = ax.imshow(mat[:,:,itrial],aspect='auto')
            plt.colorbar(ploti,ax=ax)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_title(zname + ' ' + str(itrial))

        plt.suptitle(title)
    else:
        print(['sanity check: I cannot print ' + str(N[2])+ ' subplots']) 
    return 