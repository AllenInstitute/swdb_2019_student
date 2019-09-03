def visualize_3Darray(mat,dimension = 2,dim0_label='dim 0',dim1_label = 'dim 1', dim2_label = 'dim 2',title = 'matrix',ncols=5):
    """given a numpy 3D array the function prints its size and a colorplot of each layer along the third dimension.
    This function could be used to visually inspect 3D matrices (e.g. plotting spike counts across trials and units)
    
    INPUTS mat: 2D or 3D numpy array that you would like to inspect
           dim0_label (optional): string containing the label for first [0] array's dimension
           dim1_label (optional): string containing the label for second [1] array's dimension
           dim2_label (optional): string containing the label for third [2] array's dimension
           title(optional): string containing the sup title for the plot
           ncols(optional): int number of columns to use for the subplots (default = 5)

    OUTPUTS plot the matrix using the 
    Example: inspect a matrix containing neural spike counts for 6 units along 50 time bins and 6 trials
    
    #create a random test activity matrix
    import numpy as np
    activity_matrix = np.rand.random(6,50,6)
    visualize_3Darray(activity_matrix)
    
    # if you are working in jupiter notebook and want to visualize your plot inline add the following line before calling the   function: 
    %matplotlib inline
    
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    N = mat.shape
    print(['the size of your matrix is' + str(N)])
    
    if N[dimension]>100:
        print(['only printing the first 100 subplots']) 
        
    print('plotting along dimension ' + str(dimension))
    
    if dimension==0:
        mat = np.transpose(mat,axes = np.array([1,2,0]))
        yname = dim1_label
        xname = dim2_label
        zname = dim0_label
    elif dimension==1:
        mat = np.transpose(mat,axes = np.array([2,0,1]))
        yname = dim2_label
        xname = dim0_label
        zname = dim1_label
    else:
        yname = dim0_label
        xname = dim1_label
        zname = dim2_label        
        
    N = mat.shape
    
    if len(N)==2:
        plt.imshow(mat,aspect='auto')
        plt.colorbar()
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(zname)
    else:
        if N[2]%ncols == 0:
            nrows =(N[2]//ncols)
        else:
            nrows = int(np.minimum((N[2]//ncols)+1,100/ncols))
        # plot each trial in a subplot
        fig,axes = plt.subplots(nrows,ncols, figsize=(18,nrows*4), sharex=True, sharey=True)
        axes = axes.ravel()
        axes = axes[range(np.minimum(N[2],100))]
        for itrial,ax in enumerate(axes):
            ploti = ax.imshow(mat[:,:,itrial],aspect='auto')
            plt.colorbar(ploti,ax=ax)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_title(zname + ' ' + str(itrial))

        plt.suptitle(title)
    
    return 