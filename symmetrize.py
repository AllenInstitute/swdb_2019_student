import numpy as np

def symmetrize(trMatrix):
    """ Symmetrizes upper/lower triangular similarity matrices
    
    Parameters
    ==========
    trMatrix: numpy 2D array
              triangular input matrix

    Example
    =======
    symMatrix = symmetrize(trMatrix)
    
    """
    
    return (trMatrix + trMatrix.T)/2 - np.diag(trMatrix.diagonal())

