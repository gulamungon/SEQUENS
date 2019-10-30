import numpy as np

def simultaneous_matrix_diag(A, B):
    eps    = 1e-9
    [D, V] = np.linalg.eig( B )
    D       = np.diag(1.0/np.sqrt(D + eps))  
    
    # First transform
    T1     = np.dot(D, V.T)
    
    # This should equal the identity matrix
    B1     = np.dot(np.dot(T1, B), T1.T )
    
    A1     = np.dot(np.dot(T1, A), T1.T )
    
    # Second transform is given by T2.T * (.) * T2
    [D, T2] = np.linalg.eig( A1 )
    
    # Joint transform
    T     =  np.dot(T2.T, T1)
    
    # Transform the matrices
    A2     = np.dot(np.dot(T, A), T.T )
    B2     = np.dot(np.dot(T, B), T.T )
    return T, A2, B2
